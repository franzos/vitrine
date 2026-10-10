(use-modules (guix packages)
             (guix search-paths)
             (gnu packages rust)
             (gnu packages commencement)
             (gnu packages tls)
             (gnu packages base)
             (gnu packages machine-learning)
             (gnu packages image)
             (gnu packages pkg-config)
             (gnu packages gl)
             (gnu packages freedesktop)
             (gnu packages xdisorg)
             (gnu packages vulkan))

(define openssl-with-dir
  (package
    (inherit openssl)
    (native-search-paths
     (cons (search-path-specification
            (variable "OPENSSL_DIR")
            (files '("."))
            (file-type 'directory)
            (separator #f))
           (package-native-search-paths openssl)))))

(define gcc-toolchain-with-cc
  (package
    (inherit gcc-toolchain)
    (native-search-paths
     (cons (search-path-specification
            (variable "CC")
            (files '("bin/gcc"))
            (file-type 'regular)
            (separator #f))
           (package-native-search-paths gcc-toolchain)))))

;; ort loads onnxruntime at runtime (load-dynamic) from ORT_DYLIB_PATH.
(define onnxruntime-with-dylib-path
  (package
    (inherit onnxruntime)
    (native-search-paths
     (cons (search-path-specification
            (variable "ORT_DYLIB_PATH")
            (files '("lib/libonnxruntime.so"))
            (file-type 'regular)
            (separator #f))
           (package-native-search-paths onnxruntime)))))

;; eframe dlopens libwayland-client, libxkbcommon and libEGL/libGL at runtime.
(define mesa-with-library-path
  (package
    (inherit mesa)
    (native-search-paths
     (cons (search-path-specification
            (variable "LD_LIBRARY_PATH")
            (files '("lib")))
           (package-native-search-paths mesa)))))

(packages->manifest
 (list rust-1.95
       (list rust-1.95 "cargo")
       rust-analyzer
       gcc-toolchain-with-cc
       openssl-with-dir
       pkg-config
       onnxruntime-with-dylib-path
       libjpeg-turbo-3
       libheif
       mesa-with-library-path
       wayland
       libxkbcommon
       vulkan-loader))
