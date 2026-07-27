// Synced side-by-side video comparison slider (drag to reveal)
function initVSliders() {
  document.querySelectorAll('.vslider').forEach((el) => {
    const left = el.querySelector('video.left');
    const right = el.querySelector('video.right');
    const leftLayer = el.querySelector('.left-layer');
    const handle = el.querySelector('.handle');
    const knob = el.querySelector('.knob');
    let dragging = false;

    function setPos(pct) {
      pct = Math.max(0, Math.min(100, pct));
      leftLayer.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
      handle.style.left = pct + '%';
      knob.style.left = pct + '%';
    }
    setPos(50);

    function posFromEvent(e) {
      const r = el.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      return (x / r.width) * 100;
    }

    el.addEventListener('mousedown', (e) => { e.preventDefault(); dragging = true; setPos(posFromEvent(e)); });
    el.addEventListener('touchstart', (e) => { dragging = true; setPos(posFromEvent(e)); }, { passive: true });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      if (e.buttons === 0) { dragging = false; return; } // button released outside window/element
      setPos(posFromEvent(e));
    });
    window.addEventListener('touchmove', (e) => { if (dragging) { e.preventDefault(); setPos(posFromEvent(e)); } }, { passive: false });
    window.addEventListener('mouseup', () => dragging = false);
    window.addEventListener('touchend', () => dragging = false);

    // keep both videos frame-synced
    if (left && right) {
      left.addEventListener('play', () => { try { right.currentTime = left.currentTime; right.play(); } catch (e) {} });
      right.pause();
      left.addEventListener('seeked', () => { right.currentTime = left.currentTime; });
    }
  });
}

// Play/pause videos only while visible in viewport (saves bandwidth, avoids
// dozens of videos decoding at once on load)
function initLazyPlay() {
  const videos = document.querySelectorAll('video[data-lazyplay]');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const v = entry.target;
      if (entry.isIntersecting) {
        v.play().catch(() => {});
      } else {
        v.pause();
      }
    });
  }, { threshold: 0.25 });
  videos.forEach((v) => io.observe(v));
}

// 2D image comparison: drag left/right = input vs ours, drag up/down = ours vs baseline
function initComparison2D() {
  document.querySelectorAll('.comp2d').forEach((el) => {
    const oursLayer = el.querySelector('.layer.ours');
    const inputLayer = el.querySelector('.layer.input');
    const baselineLayer = el.querySelector('.layer.baseline');
    const vDiv = el.querySelector('.v-div');
    const hDiv = el.querySelector('.h-div');
    const crosshair = el.querySelector('.crosshair');
    let dragging = false;

    // if the layers are <video>, keep them frame-synced with the input video as master
    if (inputLayer.tagName === 'VIDEO') {
      const followers = [oursLayer, baselineLayer].filter((v) => v && v.tagName === 'VIDEO');
      inputLayer.addEventListener('play', () => {
        followers.forEach((v) => { v.currentTime = inputLayer.currentTime; v.play().catch(() => {}); });
      });
      inputLayer.addEventListener('pause', () => followers.forEach((v) => v.pause()));
      inputLayer.addEventListener('seeked', () => followers.forEach((v) => v.currentTime = inputLayer.currentTime));
      inputLayer.addEventListener('timeupdate', () => {
        followers.forEach((v) => {
          if (Math.abs(v.currentTime - inputLayer.currentTime) > 0.25) v.currentTime = inputLayer.currentTime;
        });
      });
      followers.forEach((v) => v.pause());
    }

    function setPos(x, y) {
      x = Math.max(0, Math.min(100, x));
      y = Math.max(0, Math.min(100, y));
      oursLayer.style.clipPath = `inset(0 0 ${100 - y}% 0)`;
      inputLayer.style.clipPath = `inset(0 ${100 - x}% 0 0)`;
      vDiv.style.left = x + '%';
      hDiv.style.top = y + '%';
      hDiv.style.clipPath = `inset(0 0 0 ${x}%)`;
      crosshair.style.left = x + '%';
      crosshair.style.top = y + '%';
    }
    setPos(50, 50);

    function posFromEvent(e) {
      const r = el.getBoundingClientRect();
      const p = e.touches ? e.touches[0] : e;
      return { x: ((p.clientX - r.left) / r.width) * 100, y: ((p.clientY - r.top) / r.height) * 100 };
    }

    el.addEventListener('mousedown', (e) => { e.preventDefault(); dragging = true; const p = posFromEvent(e); setPos(p.x, p.y); });
    el.addEventListener('touchstart', (e) => { dragging = true; const p = posFromEvent(e); setPos(p.x, p.y); }, { passive: true });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      if (e.buttons === 0) { dragging = false; return; } // button released outside window/element
      const p = posFromEvent(e);
      setPos(p.x, p.y);
    });
    window.addEventListener('touchmove', (e) => { if (dragging) { e.preventDefault(); const p = posFromEvent(e); setPos(p.x, p.y); } }, { passive: false });
    window.addEventListener('mouseup', () => dragging = false);
    window.addEventListener('touchend', () => dragging = false);
  });

  // baseline dropdown swaps the bottom layer's image + label
  document.querySelectorAll('.comp2d-wrap').forEach((wrap) => {
    const select = wrap.querySelector('.baseline-select');
    if (!select) return;
    select.addEventListener('change', () => {
      applyBaseline(wrap);
      const target = wrap.parentElement;
      if (target && target.id) baselineIndexByTarget.set(target.id, select.selectedIndex);
    });
  });
}

// remembers the last-picked baseline dropdown index per carousel target,
// so switching examples keeps the same baseline selected instead of resetting it
const baselineIndexByTarget = new Map();

function applyBaseline(wrap) {
  const select = wrap.querySelector('.baseline-select');
  if (!select) return;
  const opt = select.selectedOptions[0];
  const baselineLayer = wrap.querySelector('.layer.baseline');
  const baselineTag = wrap.querySelector('.baseline-label');
  baselineTag.textContent = opt.dataset.label;

  if (baselineLayer.tagName === 'VIDEO') {
    const inputLayer = wrap.querySelector('.layer.input');
    baselineLayer.pause();
    baselineLayer.src = opt.value;
    baselineLayer.load();
    const onReady = () => {
      if (inputLayer) baselineLayer.currentTime = inputLayer.currentTime;
      if (inputLayer && !inputLayer.paused) baselineLayer.play().catch(() => {});
      baselineLayer.removeEventListener('loadedmetadata', onReady);
    };
    baselineLayer.addEventListener('loadedmetadata', onReady);
  } else {
    baselineLayer.src = opt.value;
  }
}

function initSceneNav() {
  document.querySelectorAll('.comp2d-carousel[data-scenes]').forEach((carousel) => {
    const target = document.querySelector(carousel.dataset.target);
    if (!target) return;
    const scenes = carousel.dataset.scenes.split(',');
    const panels = target.querySelectorAll('[data-scene]');
    let idx = 0;

    // pager dots, one dot per scene
    const progress = carousel.previousElementSibling;
    const hasProgress = progress && progress.classList.contains('scene-progress');
    let dots = [];
    if (hasProgress) {
      progress.innerHTML = '';
      scenes.forEach((_, i) => {
        const dot = document.createElement('button');
        dot.type = 'button';
        dot.className = 'progress-dot';
        dot.setAttribute('aria-label', `Go to example ${i + 1} of ${scenes.length}`);
        dot.addEventListener('click', () => show(i));
        progress.appendChild(dot);
        dots.push(dot);
      });
    }

    function show(i) {
      idx = (i + scenes.length) % scenes.length;
      panels.forEach((p) => {
        const visible = p.dataset.scene === scenes[idx];
        const isComp2d = p.classList.contains('comp2d-wrap');
        p.style.display = visible ? (isComp2d ? 'block' : 'grid') : 'none';
        if (visible && isComp2d) {
          const select = p.querySelector('.baseline-select');
          if (select && target.id && baselineIndexByTarget.has(target.id)) {
            select.selectedIndex = baselineIndexByTarget.get(target.id);
          }
          applyBaseline(p);
        }
      });
      dots.forEach((d, i) => {
        d.classList.toggle('active', i === idx);
      });
    }

    const prevBtn = carousel.querySelector('[data-nav="prev"]');
    const nextBtn = carousel.querySelector('[data-nav="next"]');
    if (prevBtn) prevBtn.addEventListener('click', () => show(idx - 1));
    if (nextBtn) nextBtn.addEventListener('click', () => show(idx + 1));
    show(0);
  });
}

function initCopyBibtex() {
  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const text = document.querySelector(btn.dataset.copyTarget).innerText;
      navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = orig, 1600);
      });
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initVSliders();
  initLazyPlay();
  initComparison2D();
  initSceneNav();
  initCopyBibtex();
});
