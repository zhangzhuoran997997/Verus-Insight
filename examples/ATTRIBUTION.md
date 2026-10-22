# Example video

`pole_cam.mp4` is **Pole cam** by [Leigh Blackall](https://www.flickr.com/photos/leighblackall),
licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

- [Original video and license](https://www.flickr.com/photos/leighblackall/2983727948).
- Obtained through [NExT-QA](https://github.com/doc-doc/NExT-QA), video ID `2983727948`.
- Verus-Bench sample: `real_nextqa_2983727948`.
- The bundled file is copied unchanged from the upstream dataset media.
- `frames/pole_cam/frame_0000.jpg` through `frame_0019.jpg` are extracted JPEG
  frames from the same video, covering a selected interval starting at 8.47 seconds.
  Their supplied timestamps are 0.00–4.75 seconds relative to that interval.
  They are copied unchanged from the benchmark frame files. Frame extraction
  and JPEG encoding are modifications of the original video.
- `--video` optionally decodes and samples the MP4 again; `--frames-json` reads
  the bundled JPEG files directly.

The video and extracted frames retain its CC BY 2.0 license. The repository's Apache-2.0 license
applies to the inference code.
