법정교육 이수현황(edu-status.html)이 스캔한 이수증을 읽을 때 쓰는 글자 인식 엔진입니다.
모두 이 사이트 안에서 불러오며, 넣은 PDF·사진은 어디로도 보내지 않습니다.
불러오지 못하면 ../ (tesseract.js)로 대신 읽습니다.

- paddle.js                         : 글자 자리 찾기(DB) → 글자 읽기(CTC) 처리 (이 사이트에서 작성)
- ort.wasm.min.js, ort-wasm-simd-threaded.mjs/.wasm
                                    : onnxruntime-web 1.20.1 (MIT, Microsoft) https://github.com/microsoft/onnxruntime
- det.onnx                          : PaddleOCR PP-OCRv4 mobile 글자 자리 찾기 (Apache-2.0)
                                      ONNX 변환본 https://huggingface.co/SWHL/RapidOCR (PP-OCRv4/ch_PP-OCRv4_det_infer.onnx)
- rec_korean.onnx, dict_korean.txt  : PaddleOCR PP-OCRv5 korean mobile 글자 읽기·글자표 (Apache-2.0)
                                      ONNX 변환본 https://huggingface.co/monkt/paddleocr-onnx (languages/korean)
  원본: PaddlePaddle/PaddleOCR https://github.com/PaddlePaddle/PaddleOCR (Apache License 2.0)
