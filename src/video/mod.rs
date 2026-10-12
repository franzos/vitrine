//! Video metadata and frames through the `ffprobe` and `ffmpeg` binaries; a process-wide permit caps concurrent runs.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::{self, Read};
use std::panic;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use image::{DynamicImage, ImageFormat};
use jiff::Timestamp;
use jiff::civil::DateTime;
use jiff::tz::TimeZone;
use serde::Deserialize;

use crate::slots::Slots;
use crate::thumbs::generate::BUCKETS;

const PROBE_TIMEOUT: Duration = Duration::from_secs(30);
const FRAME_TIMEOUT: Duration = Duration::from_secs(60);
const POLL: Duration = Duration::from_millis(10);
const STDERR_CAP: u64 = 2048;
const QUICKTIME_DATE: &str = "com.apple.quicktime.creationdate";

static RUNS: Slots = Slots::new(2);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0} not found")]
    Missing(&'static str),
    #[error("{0} timed out")]
    TimedOut(&'static str),
    #[error("{program} failed: {stderr}")]
    Failed {
        program: &'static str,
        stderr: String,
    },
    #[error("unexpected output: {0}")]
    Output(String),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoMeta {
    pub duration_ms: Option<u64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub codec: Option<String>,
    /// `%Y-%m-%dT%H:%M:%S`, local wall-clock time.
    pub taken_at: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tools {
    pub ffprobe: &'static str,
    pub ffmpeg: &'static str,
}

impl Tools {
    /// Both programs looked up on `PATH`.
    pub const SYSTEM: Self = Self {
        ffprobe: "ffprobe",
        ffmpeg: "ffmpeg",
    };

    /// `Ok` when both programs run.
    pub fn check(&self) -> Result<(), Error> {
        let never = AtomicBool::new(false);
        for program in [self.ffprobe, self.ffmpeg] {
            run(program, &[OsStr::new("-version")], PROBE_TIMEOUT, &never)?;
        }
        Ok(())
    }

    pub fn probe(&self, path: &Path, cancel: &AtomicBool) -> Result<VideoMeta, Error> {
        let args = [
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "v:0",
            "-i",
        ]
        .map(OsStr::new);
        let args: Vec<&OsStr> = args.into_iter().chain([path.as_os_str()]).collect();
        let json = run(self.ffprobe, &args, PROBE_TIMEOUT, cancel)?;
        parse(&json, &TimeZone::system())
    }

    /// One frame scaled to fit `max_side`, display rotation applied; taken at a tenth of the duration when known.
    pub fn frame(
        &self,
        path: &Path,
        duration_ms: Option<u64>,
        max_side: u32,
        cancel: &AtomicBool,
    ) -> Result<DynamicImage, Error> {
        let seek = seek_arg(duration_ms);
        let mut png = self.extract(path, seek.as_deref(), max_side, cancel)?;
        // seeking past the last keyframe of short or broken files yields nothing
        if png.is_empty() && seek.is_some() {
            png = self.extract(path, None, max_side, cancel)?;
        }
        if png.is_empty() {
            return Err(Error::Output("no frame decoded".into()));
        }
        image::load_from_memory_with_format(&png, ImageFormat::Png)
            .map_err(|e| Error::Output(e.to_string()))
    }

    /// A frame at the largest thumbnail bucket; works without a duration when probing fails.
    pub fn thumbnail_frame(&self, path: &Path, cancel: &AtomicBool) -> Result<DynamicImage, Error> {
        let duration_ms = self
            .probe(path, cancel)
            .ok()
            .and_then(|meta| meta.duration_ms);
        self.frame(path, duration_ms, BUCKETS[BUCKETS.len() - 1], cancel)
    }

    fn extract(
        &self,
        path: &Path,
        seek: Option<&str>,
        max_side: u32,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>, Error> {
        let scale = format!(
            "scale=w='min({max_side},iw)':h='min({max_side},ih)':force_original_aspect_ratio=decrease"
        );
        let mut args: Vec<&OsStr> = ["-nostdin", "-hide_banner", "-v", "error"]
            .map(OsStr::new)
            .to_vec();
        if let Some(seek) = seek {
            args.extend([OsStr::new("-ss"), OsStr::new(seek)]);
        }
        args.extend([OsStr::new("-i"), path.as_os_str()]);
        args.extend(
            [
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                &scale,
                "-f",
                "image2pipe",
                "-c:v",
                "png",
                "pipe:1",
            ]
            .map(OsStr::new),
        );
        run(self.ffmpeg, &args, FRAME_TIMEOUT, cancel)
    }
}

/// `0:07`, `1:23`, `1:02:03`.
pub fn clock(ms: u64) -> String {
    let secs = ms / 1000;
    let (hours, minutes, seconds) = (secs / 3600, secs / 60 % 60, secs % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

/// Seconds for `-ss`: a tenth of the duration, skipping fade-ins; `None` below a second.
fn seek_arg(duration_ms: Option<u64>) -> Option<String> {
    let at = duration_ms.filter(|&ms| ms >= 1000)? / 10;
    Some(format!("{}.{:03}", at / 1000, at % 1000))
}

/// Runs `program` holding a [`RUNS`] permit; stdout on a zero exit.
fn run(
    program: &'static str,
    args: &[&OsStr],
    timeout: Duration,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, Error> {
    let _permit = RUNS.acquire_unless(cancel).ok_or(Error::Cancelled)?;
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => Error::Missing(program),
            _ => Error::Io(e),
        })?;
    let mut stdout = child.stdout.take().expect("stdout is piped");
    let mut stderr = child.stderr.take().expect("stderr is piped");
    let deadline = Instant::now() + timeout;

    let out = thread::spawn(move || -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        stdout.read_to_end(&mut buf)?;
        Ok(buf)
    });
    let err = thread::spawn(move || -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        stderr.by_ref().take(STDERR_CAP).read_to_end(&mut buf)?;
        io::copy(&mut stderr, &mut io::sink())?;
        Ok(buf)
    });
    // after a kill the readers are left detached: a grandchild may still hold the pipes open
    let status = wait(&mut child, program, deadline, cancel)?;
    let out = out.join().unwrap_or_else(|p| panic::resume_unwind(p));
    let err = err.join().unwrap_or_else(|p| panic::resume_unwind(p));
    if !status.success() {
        return Err(Error::Failed {
            program,
            stderr: String::from_utf8_lossy(&err?).trim().to_owned(),
        });
    }
    Ok(out?)
}

/// Kills and reaps `child` at `deadline` or once `cancel` is set.
fn wait(
    child: &mut Child,
    program: &'static str,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<ExitStatus, Error> {
    let stop = loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(e) => break Error::Io(e),
        }
        if cancel.load(Ordering::Relaxed) {
            break Error::Cancelled;
        }
        if Instant::now() >= deadline {
            break Error::TimedOut(program);
        }
        thread::sleep(POLL);
    };
    // kill fails only when the child already exited; wait reaps it either way
    _ = child.kill();
    _ = child.wait();
    Err(stop)
}

#[derive(Deserialize)]
struct Probe {
    #[serde(default)]
    streams: Vec<Stream>,
    #[serde(default)]
    format: Format,
}

#[derive(Deserialize, Default)]
struct Format {
    duration: Option<String>,
    #[serde(default)]
    tags: HashMap<String, String>,
}

#[derive(Deserialize)]
struct Stream {
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    duration: Option<String>,
    #[serde(default)]
    side_data_list: Vec<SideData>,
}

#[derive(Deserialize)]
struct SideData {
    rotation: Option<f64>,
}

fn parse(json: &[u8], tz: &TimeZone) -> Result<VideoMeta, Error> {
    let probe: Probe = serde_json::from_slice(json).map_err(|e| Error::Output(e.to_string()))?;
    let stream = probe
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| Error::Output("no video stream".into()))?;
    let duration_ms =
        millis(probe.format.duration.as_deref()).or_else(|| millis(stream.duration.as_deref()));
    let sideways = stream
        .side_data_list
        .iter()
        .filter_map(|side| side.rotation)
        .any(|deg| (deg.rem_euclid(180.0) - 90.0).abs() < 1.0);
    let (width, height) = if sideways {
        (stream.height, stream.width)
    } else {
        (stream.width, stream.height)
    };
    Ok(VideoMeta {
        duration_ms,
        width,
        height,
        codec: stream.codec_name,
        taken_at: taken_at(&probe.format.tags, tz),
    })
}

fn millis(secs: Option<&str>) -> Option<u64> {
    let secs: f64 = secs?.parse().ok()?;
    (secs.is_finite() && secs >= 0.0).then(|| (secs * 1000.0).round() as u64)
}

/// The QuickTime date's wall-clock part as written, else `creation_time` (UTC) in `tz`; years before 1971 are placeholder dates.
fn taken_at(tags: &HashMap<String, String>, tz: &TimeZone) -> Option<String> {
    let quicktime = tags
        .get(QUICKTIME_DATE)
        .and_then(|date| date.get(..19)?.parse::<DateTime>().ok());
    let local = quicktime.or_else(|| {
        let utc: Timestamp = tags.get("creation_time")?.parse().ok()?;
        Some(tz.to_datetime(utc))
    })?;
    (local.year() >= 1971).then(|| local.strftime("%Y-%m-%dT%H:%M:%S").to_string())
}

/// Whether the system tools run; prints why a gated test is skipped when they don't.
#[cfg(test)]
pub(crate) fn tools_present() -> bool {
    match Tools::SYSTEM.check() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("skipping: {e}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use jiff::tz::offset;

    use super::*;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    fn utc() -> TimeZone {
        TimeZone::fixed(offset(0))
    }

    fn json(format: &str, stream: &str) -> Vec<u8> {
        format!(r#"{{"streams":[{{{stream}}}],"format":{{{format}}}}}"#).into_bytes()
    }

    #[test]
    fn video_parse_prefers_format_duration() {
        let meta = parse(
            &json(
                r#""duration":"7.250000""#,
                r#""codec_name":"h264","width":1920,"height":1080,"duration":"7.000000""#,
            ),
            &utc(),
        )
        .unwrap();
        assert_eq!(
            meta,
            VideoMeta {
                duration_ms: Some(7250),
                width: Some(1920),
                height: Some(1080),
                codec: Some("h264".into()),
                taken_at: None,
            }
        );
    }

    #[test]
    fn video_parse_falls_back_to_stream_duration() {
        let meta = parse(&json(r#""duration":"N/A""#, r#""duration":"2.5""#), &utc()).unwrap();
        assert_eq!(meta.duration_ms, Some(2500));
        let meta = parse(&json("", r#""duration":"0.0004""#), &utc()).unwrap();
        assert_eq!(meta.duration_ms, Some(0));
        let meta = parse(&json("", ""), &utc()).unwrap();
        assert_eq!(meta.duration_ms, None);
    }

    #[test]
    fn video_parse_swaps_dimensions_for_quarter_turns() {
        for (rotation, swapped) in [
            ("90", true),
            ("-90", true),
            ("270", true),
            ("180", false),
            ("0", false),
        ] {
            let stream = format!(
                r#""width":72,"height":56,"side_data_list":[{{"side_data_type":"Display Matrix","rotation":{rotation}}}]"#
            );
            let meta = parse(&json("", &stream), &utc()).unwrap();
            let expected = if swapped { (56, 72) } else { (72, 56) };
            assert_eq!(
                (meta.width, meta.height),
                (Some(expected.0), Some(expected.1)),
                "{rotation}"
            );
        }
    }

    #[test]
    fn video_parse_prefers_quicktime_date_wall_clock() {
        let format = r#""tags":{"creation_time":"2024-05-01T12:00:00.000000Z","com.apple.quicktime.creationdate":"2024-05-01T09:30:15+0200"}"#;
        let meta = parse(&json(format, ""), &TimeZone::fixed(offset(-5))).unwrap();
        assert_eq!(meta.taken_at.as_deref(), Some("2024-05-01T09:30:15"));
    }

    #[test]
    fn video_parse_converts_creation_time_to_local() {
        let format = r#""tags":{"creation_time":"2024-05-01T23:00:00.000000Z"}"#;
        let meta = parse(&json(format, ""), &TimeZone::fixed(offset(2))).unwrap();
        assert_eq!(meta.taken_at.as_deref(), Some("2024-05-02T01:00:00"));
        let meta = parse(&json(format, ""), &utc()).unwrap();
        assert_eq!(meta.taken_at.as_deref(), Some("2024-05-01T23:00:00"));
    }

    #[test]
    fn video_parse_drops_dates_before_1971() {
        for tags in [
            r#""tags":{"creation_time":"1970-01-01T00:00:00.000000Z"}"#,
            r#""tags":{"creation_time":"1904-01-01T00:00:00.000000Z"}"#,
            r#""tags":{"com.apple.quicktime.creationdate":"1970-06-01T00:00:00+0000"}"#,
            r#""tags":{"creation_time":"garbage"}"#,
        ] {
            assert_eq!(
                parse(&json(tags, ""), &utc()).unwrap().taken_at,
                None,
                "{tags}"
            );
        }
    }

    #[test]
    fn video_parse_without_a_video_stream_fails() {
        let err = parse(br#"{"streams":[],"format":{"duration":"1.0"}}"#, &utc()).unwrap_err();
        assert!(matches!(err, Error::Output(_)), "{err:?}");
        assert!(matches!(parse(b"not json", &utc()), Err(Error::Output(_))));
    }

    #[test]
    fn video_seek_arg_is_a_tenth_from_one_second() {
        assert_eq!(seek_arg(None), None);
        assert_eq!(seek_arg(Some(999)), None);
        assert_eq!(seek_arg(Some(1000)).as_deref(), Some("0.100"));
        assert_eq!(seek_arg(Some(73_456)).as_deref(), Some("7.345"));
    }

    #[test]
    fn video_clock_formats() {
        assert_eq!(clock(0), "0:00");
        assert_eq!(clock(7_999), "0:07");
        assert_eq!(clock(83_000), "1:23");
        assert_eq!(clock(3_599_000), "59:59");
        assert_eq!(clock(3_723_000), "1:02:03");
    }

    fn sh(script: &str, timeout: Duration, cancel: &AtomicBool) -> Result<Vec<u8>, Error> {
        run(
            "sh",
            &[OsStr::new("-c"), OsStr::new(script)],
            timeout,
            cancel,
        )
    }

    #[test]
    fn video_run_kills_at_the_deadline() {
        let start = Instant::now();
        let err = sh(
            "sleep 30",
            Duration::from_millis(200),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(matches!(err, Error::TimedOut("sh")), "{err:?}");
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "{:?}",
            start.elapsed()
        );
    }

    #[test]
    fn video_run_stops_when_cancelled() {
        let cancel = AtomicBool::new(false);
        let start = Instant::now();
        thread::scope(|s| {
            let job = s.spawn(|| sh("sleep 30", Duration::from_secs(30), &cancel));
            thread::sleep(Duration::from_millis(100));
            cancel.store(true, Ordering::Relaxed);
            assert!(matches!(job.join().unwrap(), Err(Error::Cancelled)));
        });
        assert!(start.elapsed() < Duration::from_secs(3));
        assert!(matches!(
            sh("true", Duration::from_secs(5), &cancel),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn video_run_drains_large_output() {
        let out = sh(
            "head -c 1048576 /dev/zero; head -c 1048576 /dev/zero >&2",
            Duration::from_secs(10),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(out.len(), 1 << 20);
    }

    #[test]
    fn video_run_returns_stderr_on_failure() {
        let err = sh(
            "echo partial; echo boom >&2; exit 3",
            Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        match err {
            Error::Failed { program, stderr } => {
                assert_eq!(program, "sh");
                assert_eq!(stderr, "boom");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn video_missing_tools_report_which() {
        let tools = Tools {
            ffprobe: "/nonexistent/ffprobe",
            ffmpeg: "/nonexistent/ffmpeg",
        };
        let never = AtomicBool::new(false);
        let clip = fixture("clip.mp4");
        assert!(matches!(
            tools.check(),
            Err(Error::Missing("/nonexistent/ffprobe"))
        ));
        assert!(matches!(
            tools.probe(&clip, &never),
            Err(Error::Missing("/nonexistent/ffprobe"))
        ));
        assert!(matches!(
            tools.frame(&clip, Some(1000), 256, &never),
            Err(Error::Missing("/nonexistent/ffmpeg"))
        ));
        assert!(matches!(
            tools.thumbnail_frame(&clip, &never),
            Err(Error::Missing("/nonexistent/ffmpeg"))
        ));
    }

    #[test]
    fn video_probe_reads_the_fixtures() {
        if !tools_present() {
            return;
        }
        let never = AtomicBool::new(false);
        let meta = Tools::SYSTEM.probe(&fixture("clip.mp4"), &never).unwrap();
        assert_eq!(meta.duration_ms, Some(1000));
        assert_eq!((meta.width, meta.height), (Some(72), Some(56)));
        assert_eq!(meta.codec.as_deref(), Some("h264"));
        assert!(
            meta.taken_at.as_deref().unwrap().starts_with("2024-05-0"),
            "{meta:?}"
        );
        let rotated = Tools::SYSTEM
            .probe(&fixture("clip-rotated.mp4"), &never)
            .unwrap();
        assert_eq!((rotated.width, rotated.height), (Some(56), Some(72)));
    }

    #[test]
    fn video_frame_fits_and_rotates() {
        if !tools_present() {
            return;
        }
        let never = AtomicBool::new(false);
        let frame = Tools::SYSTEM
            .frame(&fixture("clip.mp4"), Some(1000), 32, &never)
            .unwrap();
        assert!(frame.width().max(frame.height()) <= 32);
        let frame = Tools::SYSTEM
            .thumbnail_frame(&fixture("clip.mp4"), &never)
            .unwrap();
        assert_eq!((frame.width(), frame.height()), (72, 56));
        assert!(frame.width().max(frame.height()) <= 512);
        let rotated = Tools::SYSTEM
            .frame(&fixture("clip-rotated.mp4"), None, 512, &never)
            .unwrap();
        assert_eq!((rotated.width(), rotated.height()), (56, 72));
    }
}
