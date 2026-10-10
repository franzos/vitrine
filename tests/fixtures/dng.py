"""Writes a minimal DNG: no raw data, only the tags quickraw reads and a JPEG preview in the second SubIFD."""

import struct
import sys

BYTE, ASCII, SHORT, LONG, SRATIONAL = 1, 2, 3, 4, 10
SIZES = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, SRATIONAL: 8}


def ifd_size(entries):
    return 2 + 12 * len(entries) + 4


def write_ifd(out, entries, data_at):
    """Appends an IFD to `out`. Returns the values over 4 bytes, which the IFD expects at `data_at`."""
    out += struct.pack("<H", len(entries))
    extra = b""
    for tag, kind, count, value in sorted(entries):
        raw = value if isinstance(value, bytes) else b""
        if not raw:
            raw = struct.pack({SHORT: "<H", LONG: "<I"}[kind], value)
        if len(raw) <= 4:
            out += struct.pack("<HHI", tag, kind, count) + raw.ljust(4, b"\0")
        else:
            out += struct.pack("<HHII", tag, kind, count, data_at + len(extra))
            extra += raw + (b"\0" if len(raw) % 2 else b"")
    out += struct.pack("<I", 0)
    return extra


def ascii(text):
    raw = text.encode() + b"\0"
    return (ASCII, len(raw), raw)


def main(preview_path, out_path, orientation=1):
    jpeg = open(preview_path, "rb").read()
    width, height = jpeg_size(jpeg)
    identity = b"".join(struct.pack("<ii", int(i % 4 == 0), 1) for i in range(9))

    ifd0_at = 8
    ifd0 = [
        (0x010F, *ascii("vitrine")),
        (0x0110, *ascii("fixture camera")),
        (0x0112, SHORT, 1, orientation),
        (0x014A, LONG, 2, bytes(8)),
        (0xC612, BYTE, 4, bytes([1, 4, 0, 0])),
        (0xC614, *ascii("vitrine fixture camera")),
        (0xC622, SRATIONAL, 9, identity),
    ]
    raw_ifd = [(0x00FE, LONG, 1, 0)]
    preview_ifd = [
        (0x00FE, LONG, 1, 1),
        (0x0100, LONG, 1, width),
        (0x0101, LONG, 1, height),
        (0x0103, SHORT, 1, 7),
        (0x0111, LONG, 1, 0),
        (0x0117, LONG, 1, len(jpeg)),
    ]

    # layout: header, IFD0, its data (incl. SubIFD offsets), raw IFD, preview IFD, JPEG
    raw_at = ifd0_at + ifd_size(ifd0) + len(write_ifd(bytearray(), ifd0, 0))
    preview_at = raw_at + ifd_size(raw_ifd)
    jpeg_at = preview_at + ifd_size(preview_ifd)

    ifd0[3] = (0x014A, LONG, 2, struct.pack("<II", raw_at, preview_at))
    preview_ifd[4] = (0x0111, LONG, 1, jpeg_at)

    out = bytearray(b"II" + struct.pack("<HI", 42, ifd0_at))
    extra = write_ifd(out, ifd0, ifd0_at + ifd_size(ifd0))
    out += extra
    assert len(out) == raw_at
    out += write_ifd(out, raw_ifd, 0)
    out += write_ifd(out, preview_ifd, 0)
    assert len(out) == jpeg_at
    out += jpeg
    open(out_path, "wb").write(out)


def jpeg_size(data):
    i = 2
    while i < len(data):
        marker, length = data[i + 1], struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        i += 2 + length
    raise ValueError("no SOF marker")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:]))
