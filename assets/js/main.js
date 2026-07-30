

// Play/pause teaser videos only while visible in viewport
function initLazyPlay() {
  const videos = document.querySelectorAll('.teaser-video');
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

// Global observer for pausing videos when they are not in the viewport
const compViewportObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.dispatchEvent(new CustomEvent('viewport-enter'));
    } else {
      entry.target.dispatchEvent(new CustomEvent('viewport-leave'));
    }
  });
}, { threshold: 0.1 });

// Keep the three visible video layers on the same decoded frame. Seeking is
// limited to panel/baseline changes and resume; seeking on every timeupdate
// makes comparison widgets visibly jitter.
class VideoSyncManager {
  constructor(container, videos) {
    this.container = container;
    this.videos = Array.from(videos);
    this.master = this.videos.find(v => v.classList.contains('input')) || this.videos[0];
    this.isReady = false;
    this.isInViewport = false;
    this.destroyed = false;
    this.starting = false;
    this.videos.forEach(v => { v.loop = true; v.preload = 'auto'; });
    this.onCanPlayHandler = this.onCanPlay.bind(this);
    this.onViewportEnterHandler = this.onViewportEnter.bind(this);
    this.onViewportLeaveHandler = this.onViewportLeave.bind(this);
    this.init();
  }

  init() {
    this.videos.forEach(v => v.addEventListener('canplay', this.onCanPlayHandler));
    this.container.addEventListener('viewport-enter', this.onViewportEnterHandler);
    this.container.addEventListener('viewport-leave', this.onViewportLeaveHandler);
    compViewportObserver.observe(this.container);
    this.checkReady();
  }

  onCanPlay() { this.checkReady(); }

  checkReady() {
    if (this.isReady) {
      if (this.isInViewport && this.master.paused) this.startAll();
      return;
    }
    if (this.videos.every(v => v.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA)) {
      this.isReady = true;
      if (this.isInViewport) this.startAll(true);
    }
  }

  seek(video, time) {
    if (Math.abs(video.currentTime - time) < 0.01) return Promise.resolve();
    return new Promise((resolve) => {
      video.addEventListener('seeked', resolve, { once: true });
      video.currentTime = time;
    });
  }

  async startAll(forceSync = false) {
    if (!this.isReady || this.starting || this.destroyed) return;
    this.starting = true;
    const time = this.master.currentTime || 0;
    const outOfSync = forceSync
      ? this.videos
      : this.videos.filter(v => Math.abs(v.currentTime - time) > 0.04);
    this.videos.forEach(v => v.pause());
    await Promise.all(outOfSync.map(v => this.seek(v, time)));
    if (!this.isInViewport || this.destroyed) {
      this.starting = false;
      return;
    }
    await Promise.all(this.videos.map(v => v.play().catch(err => {
      if (err.name !== 'AbortError') console.warn('Video play error:', err);
    })));
    this.starting = false;
  }

  pauseAll() { this.videos.forEach(v => v.pause()); }
  onViewportEnter() { this.isInViewport = true; this.checkReady(); }
  onViewportLeave() { this.isInViewport = false; this.pauseAll(); }

  destroy() {
    this.destroyed = true;
    compViewportObserver.unobserve(this.container);
    this.videos.forEach(v => { v.removeEventListener('canplay', this.onCanPlayHandler); v.pause(); });
    this.container.removeEventListener('viewport-enter', this.onViewportEnterHandler);
    this.container.removeEventListener('viewport-leave', this.onViewportLeaveHandler);
  }
}

const activeSyncManagers = new Map();

function syncActivePanelVideos(targetId, activePanel) {
  activeSyncManagers.get(targetId)?.destroy();
  const comp = activePanel.querySelector('.comp2d');
  const videos = comp?.querySelectorAll('video');
  if (!videos?.length) return;
  activeSyncManagers.set(targetId, new VideoSyncManager(comp, videos));
}

// 2D comparison: drag on the image or handle to reveal all three layers.
function initComparison2D() {
  document.querySelectorAll('.comp2d').forEach((el) => {
    let cachedRect = null;
    let rafId = null;
    let pendingX = 50, pendingY = 50;

    function setPos(x, y) {
      x = Math.max(0, Math.min(100, x));
      y = Math.max(0, Math.min(100, y));
      // Two properties fan out to every layer in CSS, avoiding six independent
      // style updates on every pointer frame.
      el.style.setProperty('--compare-x', `${x}%`);
      el.style.setProperty('--compare-y', `${y}%`);
    }

    function posFromEvent(e) {
      if (!cachedRect) cachedRect = el.getBoundingClientRect();
      return {
        x: ((e.clientX - cachedRect.left) / cachedRect.width) * 100,
        y: ((e.clientY - cachedRect.top) / cachedRect.height) * 100
      };
    }

    function scheduleSetPos(x, y) {
      pendingX = x;
      pendingY = y;
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          setPos(pendingX, pendingY);
          rafId = null;
        });
      }
    }

    function startDrag(e) {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      el.classList.add('is-dragging');
      cachedRect = el.getBoundingClientRect();
      const p = posFromEvent(e);
      scheduleSetPos(p.x, p.y);
    }

    function moveDrag(e) {
      if (!el.hasPointerCapture(e.pointerId)) return;
      const p = posFromEvent(e);
      scheduleSetPos(p.x, p.y);
    }

    function endDrag(e) {
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
      el.classList.remove('is-dragging');
      cachedRect = null;
    }

    el.addEventListener('pointerdown', startDrag);
    el.addEventListener('pointermove', moveDrag);
    el.addEventListener('pointerup', endDrag);
    el.addEventListener('pointercancel', endDrag);
    window.addEventListener('resize', () => { cachedRect = null; });
  });

  document.querySelectorAll('.comp2d-wrap').forEach((wrap) => {
    const select = wrap.querySelector('.baseline-select');
    if (!select) return;
    select.addEventListener('change', () => {
      applyBaseline(wrap);
      const target = wrap.parentElement;
      if (target?.id) {
        baselineIndexByTarget.set(target.id, select.selectedIndex);
        syncActivePanelVideos(target.id, wrap);
      }
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
    baselineLayer.pause();
    baselineLayer.src = opt.value;
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
      let activePanel = null;
      panels.forEach((p) => {
        const visible = p.dataset.scene === scenes[idx];
        const isComp2d = p.classList.contains('comp2d-wrap');
        p.style.display = visible ? (isComp2d ? 'block' : 'grid') : 'none';
        if (visible) {
          activePanel = p;
          if (isComp2d) {
            const select = p.querySelector('.baseline-select');
            if (select && target.id && baselineIndexByTarget.has(target.id)) {
              select.selectedIndex = baselineIndexByTarget.get(target.id);
            }
            applyBaseline(p);
          }
        }
      });
      dots.forEach((d, i) => {
        d.classList.toggle('active', i === idx);
      });

      // Synchronize videos in the newly shown active panel
      if (activePanel && target.id) {
        syncActivePanelVideos(target.id, activePanel);
      }
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
  initLazyPlay();
  initComparison2D();
  initSceneNav();
  initCopyBibtex();
});
