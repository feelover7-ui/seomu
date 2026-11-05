/* PaddleOCR 한국어 글자 인식 (이 브라우저 안에서만)
   - 글자 자리 찾기: PP-OCRv4 mobile det (det.onnx)
   - 글자 읽기    : PP-OCRv5 korean mobile rec (rec_korean.onnx, dict_korean.txt)
   - 실행         : onnxruntime-web (wasm, 한 줄기)
   PADDLE.load(base) → Promise, PADDLE.read(canvas) → Promise<[{text, conf, x0, y0, x1, y1}]> (canvas 좌표) */
(function () {
  var base = '', ready = null, det = null, rec = null, dict = null;

  function load(b) {
    if (ready) return ready;
    base = b;
    ready = new Promise(function (ok, no) {
      if (window.ort) return ok();
      var s = document.createElement('script'); s.src = base + 'ort.wasm.min.js';
      s.onload = function () { ok(); }; s.onerror = function () { no(new Error('글자 인식 엔진 파일을 불러오지 못함')); };
      document.head.appendChild(s);
    }).then(function () {
      ort.env.wasm.wasmPaths = base;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      var opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      return Promise.all([
        ort.InferenceSession.create(base + 'det.onnx', opt),
        ort.InferenceSession.create(base + 'rec_korean.onnx', opt),
        fetch(base + 'dict_korean.txt').then(function (r) { if (!r.ok) throw new Error('글자표를 불러오지 못함'); return r.text(); })
      ]);
    }).then(function (r) {
      det = r[0]; rec = r[1];
      dict = [''].concat(r[2].replace(/\r/g, '').split('\n').filter(function (x, i, a) { return !(i === a.length - 1 && x === ''); }), [' ']);
    }).catch(function (e) { ready = null; throw e; });
    return ready;
  }

  /* 1) 글자 자리: DB 확률 지도 → 이어진 덩어리 → 상자 (기울지 않은 문서라 가로세로 상자) */
  function detect(src) {
    var W = src.width, H = src.height, lim = 1600;
    var r = Math.min(1, lim / Math.max(W, H));
    var rw = Math.max(32, Math.round(W * r / 32) * 32), rh = Math.max(32, Math.round(H * r / 32) * 32);
    var c = document.createElement('canvas'); c.width = rw; c.height = rh;
    var x = c.getContext('2d'); x.drawImage(src, 0, 0, rw, rh);
    var px = x.getImageData(0, 0, rw, rh).data, n = rw * rh, data = new Float32Array(3 * n);
    var mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      data[i] = (px[j] / 255 - mean[0]) / std[0];
      data[n + i] = (px[j + 1] / 255 - mean[1]) / std[1];
      data[2 * n + i] = (px[j + 2] / 255 - mean[2]) / std[2];
    }
    var feeds = {}; feeds[det.inputNames[0]] = new ort.Tensor('float32', data, [1, 3, rh, rw]);
    return det.run(feeds).then(function (o) {
      var pm = o[det.outputNames[0]].data, seen = new Uint8Array(n), boxes = [], stack = new Int32Array(n);
      for (var p = 0; p < n; p++) {
        if (seen[p] || pm[p] <= 0.3) continue;
        var top = 0, minx = rw, maxx = 0, miny = rh, maxy = 0, sum = 0, cnt = 0;
        stack[top++] = p; seen[p] = 1;
        while (top) {
          var q = stack[--top], qx = q % rw, qy = (q / rw) | 0;
          sum += pm[q]; cnt++;
          if (qx < minx) minx = qx; if (qx > maxx) maxx = qx; if (qy < miny) miny = qy; if (qy > maxy) maxy = qy;
          if (qx > 0 && !seen[q - 1] && pm[q - 1] > 0.3) { seen[q - 1] = 1; stack[top++] = q - 1; }
          if (qx < rw - 1 && !seen[q + 1] && pm[q + 1] > 0.3) { seen[q + 1] = 1; stack[top++] = q + 1; }
          if (qy > 0 && !seen[q - rw] && pm[q - rw] > 0.3) { seen[q - rw] = 1; stack[top++] = q - rw; }
          if (qy < rh - 1 && !seen[q + rw] && pm[q + rw] > 0.3) { seen[q + rw] = 1; stack[top++] = q + rw; }
        }
        var bw = maxx - minx + 1, bh = maxy - miny + 1;
        if (bw < 3 || bh < 3 || sum / cnt < 0.6) continue;
        var d = bw * bh * 1.5 / (2 * (bw + bh));         // unclip (DB 1.5배)
        var sx = W / rw, sy = H / rh;
        boxes.push({ x0: Math.max(0, (minx - d) * sx), y0: Math.max(0, (miny - d) * sy),
                     x1: Math.min(W, (maxx + 1 + d) * sx), y1: Math.min(H, (maxy + 1 + d) * sy) });
      }
      return boxes;
    });
  }

  /* 2) 글자 읽기: 높이 48로 맞춰 CTC */
  function readBox(src, b) {
    var bw = b.x1 - b.x0, bh = b.y1 - b.y0;
    if (bw < 2 || bh < 2) return Promise.resolve(null);
    var nw = Math.max(8, Math.ceil(48 * bw / bh));
    var c = document.createElement('canvas'); c.width = nw; c.height = 48;
    var x = c.getContext('2d'); x.imageSmoothingQuality = 'high';
    x.drawImage(src, b.x0, b.y0, bw, bh, 0, 0, nw, 48);
    var px = x.getImageData(0, 0, nw, 48).data, n = nw * 48, data = new Float32Array(3 * n);
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      data[i] = px[j] / 127.5 - 1; data[n + i] = px[j + 1] / 127.5 - 1; data[2 * n + i] = px[j + 2] / 127.5 - 1;
    }
    var feeds = {}; feeds[rec.inputNames[0]] = new ort.Tensor('float32', data, [1, 3, 48, nw]);
    return rec.run(feeds).then(function (o) {
      var t = o[rec.outputNames[0]], T = t.dims[1], C = t.dims[2], v = t.data, out = '', last = -1, ps = 0, k = 0;
      for (var s = 0; s < T; s++) {
        var bi = 0, bv = -1, off = s * C;
        for (var cc = 0; cc < C; cc++) if (v[off + cc] > bv) { bv = v[off + cc]; bi = cc; }
        if (bi !== last && bi !== 0) { out += dict[bi] || ''; ps += bv; k++; }
        last = bi;
      }
      return out ? { text: out, conf: k ? ps / k : 0, x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 } : null;
    });
  }

  function read(src) {
    return detect(src).then(function (boxes) {
      var res = [], chain = Promise.resolve();
      boxes.forEach(function (b) { chain = chain.then(function () { return readBox(src, b); }).then(function (r) { if (r) res.push(r); }); });
      return chain.then(function () { return res; });
    });
  }

  window.PADDLE = { load: load, read: read };
})();
