/* ── Celesta · app.js ─────────────────────────────────────────────────── */
/* All features work fully client-side. No backend required.             */

/* ── 1. Three.js Starfield (lazy-init on idle) ────────────────────────── */
(function () {
  let renderer, scene, camera, stars, sky, planet, orbs = [];
  let mouse = { x: 0, y: 0 }, targetMouse = { x: 0, y: 0 };
  let t = 0, running = false, initDone = false;

  function initThree() {
    if (initDone) return;
    initDone = true;
    try {
      const canvas = document.getElementById('starfield');
      if (!canvas) return;
      renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.2));
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setClearColor(0x000000, 0);

      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
      camera.position.z = 600;

      const starCount = 1200;
      const positions = new Float32Array(starCount * 3);
      for (let i = 0; i < starCount; i++) {
        positions[i * 3]     = (Math.random() - 0.5) * 2000;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 2000;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 2000;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const mat = new THREE.PointsMaterial({ color: 0xaaaacc, size: 1.0, sizeAttenuation: true, transparent: true, opacity: 0.6 });
      stars = new THREE.Points(geo, mat);
      scene.add(stars);

      const loader = new THREE.TextureLoader();
      const skyTex = loader.load('/static/images/textures/starfield_milkyway.jpg');
      sky = new THREE.Mesh(
        new THREE.SphereGeometry(950, 24, 24),
        new THREE.MeshBasicMaterial({ map: skyTex, side: THREE.BackSide, transparent: true, opacity: 0.5 })
      );
      scene.add(sky);

      const planetTex = loader.load('/static/images/textures/hero_planet.jpg');
      planet = new THREE.Mesh(
        new THREE.SphereGeometry(90, 36, 36),
        new THREE.MeshBasicMaterial({ map: planetTex })
      );
      planet.position.set(320, -60, -260);
      scene.add(planet);

      scene.add(new THREE.AmbientLight(0x6b7bb8, 0.5));

      const addOrb = (x, y, z, color, sz) => {
        const g = new THREE.SphereGeometry(sz, 8, 8);
        const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.03 });
        const mesh = new THREE.Mesh(g, m);
        mesh.position.set(x, y, z);
        scene.add(mesh);
        return mesh;
      };
      orbs = [
        addOrb(-300, 200, -300, 0x6366f1, 200),
        addOrb(300, -150, -400, 0x06b6d4, 160),
        addOrb(100, 300, -200, 0x8b5cf6, 130),
      ];

      document.addEventListener('mousemove', e => {
        targetMouse.x = (e.clientX / window.innerWidth - 0.5) * 2;
        targetMouse.y = -(e.clientY / window.innerHeight - 0.5) * 2;
      });

      running = true;
      animate();
    } catch(e) {}
  }

  function animate() {
    if (!running) return;
    requestAnimationFrame(animate);
    t += 0.0004;
    mouse.x += (targetMouse.x - mouse.x) * 0.04;
    mouse.y += (targetMouse.y - mouse.y) * 0.04;
    stars.rotation.x = t * 0.4 + mouse.y * 0.06;
    stars.rotation.y = t * 0.6 + mouse.x * 0.06;
    sky.rotation.y = t * 0.05;
    planet.rotation.y = t * 2.2;
    planet.position.x = 320 + mouse.x * 12;
    planet.position.y = -180 + mouse.y * 8;
    orbs.forEach((o, i) => {
      o.position.y += Math.sin(t * 0.8 + i * 2.1) * 0.15;
      o.rotation.z = t * 0.2 + i;
    });
    renderer.render(scene, camera);
  }

  window.addEventListener('resize', () => {
    if (!camera) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  if ('requestIdleCallback' in window) {
    requestIdleCallback(initThree, { timeout: 800 });
  } else {
    setTimeout(initThree, 400);
  }
})();

/* ── 1b. Mobile nav toggle ─────────────────────────────────────────────── */
(function () {
  const toggle = document.getElementById('nav-toggle');
  const links = document.getElementById('nav-links');
  if (!toggle || !links) return;
  function closeMenu() { toggle.classList.remove('open'); links.classList.remove('open'); toggle.setAttribute('aria-expanded', 'false'); }
  function openMenu() { toggle.classList.add('open'); links.classList.add('open'); toggle.setAttribute('aria-expanded', 'true'); }
  toggle.addEventListener('click', () => { links.classList.contains('open') ? closeMenu() : openMenu(); });
  links.querySelectorAll('a').forEach(a => a.addEventListener('click', closeMenu));
  window.addEventListener('resize', () => { if (window.innerWidth > 768) closeMenu(); });
})();

/* ── 2. Reveal on scroll ───────────────────────────────────────────────── */
(function () {
  const els = document.querySelectorAll('.reveal');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry, idx) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), idx * 40);
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  els.forEach(el => io.observe(el));
})();

/* ── 3. Animated number counters ──────────────────────────────────────── */
(function () {
  const counters = document.querySelectorAll('.stat-num[data-target]');
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      io.unobserve(entry.target);
      const el = entry.target;
      const target = parseFloat(el.dataset.target);
      const isFloat = String(target).includes('.');
      const decimals = isFloat ? 1 : 0;
      const duration = 1200;
      const start = performance.now();
      function tick(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = eased * target;
        el.textContent = isFloat ? value.toFixed(decimals) : Math.round(value).toLocaleString();
        if (progress < 1) requestAnimationFrame(tick);
        else el.textContent = isFloat ? target.toFixed(decimals) : Math.round(target).toLocaleString();
      }
      requestAnimationFrame(tick);
    });
  }, { threshold: 0.5 });
  counters.forEach(c => io.observe(c));
})();

/* ── 4. Animate bars on scroll ────────────────────────────────────────── */
(function () {
  const bars = document.querySelectorAll('.metric-fill, .feat-bar');
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.style.width = entry.target.style.width; io.unobserve(entry.target); }
    });
  }, { threshold: 0.3 });
  bars.forEach(b => io.observe(b));
})();

/* ── 5. Confusion Matrix ───────────────────────────────────────────────── */
(function () {
  const stats = window.CELESTA_STATS;
  if (!stats || !stats.confusion_matrix) return;
  const cm = stats.confusion_matrix;
  const classes = stats.classes;
  const n = classes.length;
  const grid = document.getElementById('cm-grid');
  if (!grid) return;
  grid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  grid.style.maxWidth = `${n * 130}px`;
  const maxVal = Math.max(...cm.flat());
  cm.forEach((row, r) => {
    row.forEach((val, c) => {
      const cell = document.createElement('div');
      cell.className = `cm-cell ${r === c ? 'diagonal' : 'off-diagonal'}`;
      const intensity = val / maxVal;
      if (r !== c) {
        cell.style.background = `rgba(239,68,68,${intensity * 0.25})`;
        cell.style.borderColor = `rgba(239,68,68,${intensity * 0.3})`;
      }
      cell.innerHTML = `${val}<small>${classes[r]}→${classes[c]}</small>`;
      grid.appendChild(cell);
    });
  });
})();

/* ── 6. Explore section (client-side Wikipedia + instant images) ──────── */
(function () {
  const input      = document.getElementById('explore-input');
  const dropdown   = document.getElementById('explore-dropdown');
  const clearBtn   = document.getElementById('explore-clear-btn');
  const card       = document.getElementById('obj-card');
  const loading    = document.getElementById('obj-loading');
  const content    = document.getElementById('obj-content');
  const imgEl      = document.getElementById('obj-image');
  const imgWrap    = document.getElementById('obj-image-wrap');
  const badgeEl    = document.getElementById('obj-badge');
  const titleEl    = document.getElementById('obj-title');
  const extractEl  = document.getElementById('obj-extract');
  const linksEl    = document.getElementById('obj-links');

  if (!input) return;

  let debounceTimer = null;
  let searchSeq = 0;
  let detailsSeq = 0;
  const _detailCache = new Map();
  const _searchCache = new Map();

  const KNOWN_IMAGES = {
    // Exoplanets — curated from NASA, ESA, and science sources
    'Kepler-22b': 'https://assets.science.nasa.gov/content/dam/science/astro/exo-explore/2024/03/Kepler22b.jpg/jcr:content/renditions/cq5dam.web.1280.1280.jpeg',
    'Proxima Centauri b': 'https://assets.science.nasa.gov/dynamicimage/assets/science/astro/exo-explore/assets/content/planets/superearth-7.jpg?w=1280&fit=clip',
    'TRAPPIST-1e': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/TRAPPIST-1e_artist_impression_2018.jpg/640px-TRAPPIST-1e_artist_impression_2018.jpg',
    '51 Pegasi b': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Artist%27s_impression_of_51_Pegasi_b.jpg/640px-Artist%27s_impression_of_51_Pegasi_b.jpg',
    'HD 209458 b': 'https://images-assets.nasa.gov/image/PIA20056/PIA20056~medium.jpg',
    'Kepler-452b': 'https://images-assets.nasa.gov/image/PIA19825/PIA19825~medium.jpg',
    // Stars — Hubble/ESO imagery
    'Proxima Centauri': 'https://astrobiology.nasa.gov/uploads/filer_public/b6/ea/b6ea3f72-e29b-4a82-a7a1-c9bdc0c1d59d/artists_impression_of_proxima_centauri_b_shown_hypothetically_as_an_arid_rocky_super-earth.jpg',
    'Betelgeuse': 'https://cdn.eso.org/images/screen/eso0927a.jpg',
    'Sirius': 'https://assets.science.nasa.gov/dynamicimage/assets/science/missions/hubble/releases/2005/12/STScI-01EVT7Z416PH8S7BGJQZH87ARV.tif?w=1280&fit=clip',
    'Alpha Centauri': 'https://c02.purpledshub.com/uploads/sites/48/2025/08/alpha-centauri-a-planet-illustration.jpg',
    'Polaris': 'https://assets.science.nasa.gov/dynamicimage/assets/science/missions/hubble/releases/2006/01/STScI-01EVT7S2CEJ2N8QJ1RDM6BZRC0.tif?w=1280&fit=clip',
    // Nebulae
    'Orion Nebula': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Orion_Nebula_-_Hubble_2006.jpg/640px-Orion_Nebula_-_Hubble_2006.jpg',
    'Crab Nebula': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Crab_Nebula.jpg/640px-Crab_Nebula.jpg',
    'Eagle Nebula': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Eagle_nebula_pillars.jpg/640px-Eagle_nebula_pillars.jpg',
    'Helix Nebula': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Helix_Nebula_%28NGS%29.jpg/640px-Helix_Nebula_%28NGS%29.jpg',
    'Pillars of Creation': 'https://assets.science.nasa.gov/dynamicimage/assets/science/missions/webb/science/2022/10/STScI-01GGF8H15VZ09MET9HFBRQX4S3.png?w=1280&fit=clip',
    // Galaxies
    'Andromeda Galaxy': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Andromeda_Galaxy_%28with_h-alpha%29.jpg/640px-Andromeda_Galaxy_%28with_h-alpha%29.jpg',
    'Triangulum Galaxy': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Triangulum_Galaxy_by_GALEX.jpg/640px-Triangulum_Galaxy_by_GALEX.jpg',
    'Milky Way': 'https://www.universetoday.com/article_images/milky_way.jpg',
    'Whirlpool Galaxy': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/M51_whirlpool_galaxy.jpg/640px-M51_whirlpool_galaxy.jpg',
  };

  const PLACEHOLDERS = { Exoplanet: '🌍', Galaxy: '🌀', Nebula: '🌌', Star: '⭐', Planet: '🪐', Moon: '🌙', 'Star Cluster': '✨', 'Celestial Object': '🔭' };

  function inferBadge(description, title) {
    const d = (description + ' ' + title).toLowerCase();
    if (d.includes('exoplanet') || d.includes('extrasolar planet')) return 'Exoplanet';
    if (d.includes('galaxy')) return 'Galaxy';
    if (d.includes('nebula')) return 'Nebula';
    if (d.includes('cluster')) return 'Star Cluster';
    if (d.includes('star') || d.includes('dwarf') || d.includes('giant')) return 'Star';
    if (d.includes('planet')) return 'Planet';
    if (d.includes('moon') || d.includes('satellite')) return 'Moon';
    return 'Celestial Object';
  }

  function showImage(url, fallbacks) {
    const queue = (fallbacks || []).filter(u => u && u !== url);
    imgEl.classList.remove('loaded');
    imgEl.onload = () => imgEl.classList.add('loaded');
    imgEl.onerror = () => {
      // This image link is dead/broken — try the next candidate, if any,
      // instead of leaving a permanently blank image.
      const next = queue.shift();
      if (next) { imgEl.src = next; } else { imgEl.removeAttribute('src'); }
    };
    imgEl.src = url;
  }

  /* ── Search: backend proxies to Wikipedia ───────────────────────────── */
  async function wikiSearch(q) {
    const url = `/api/explore/search?q=${encodeURIComponent(q)}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(6000) });
    if (!res.ok) throw new Error('Search failed');
    const data = await res.json();
    return Array.isArray(data) ? data : [];
  }

  /* ── Details: backend proxies to Google Custom Search image, NASA     */
  /* Images API, and Wikipedia — Google + NASA take priority over Wiki.  */
  async function objectDetails(name) {
    const url = `/api/explore/details?name=${encodeURIComponent(name)}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) throw new Error('Not found');
    return await res.json();
  }

  function renderObject(name, data) {
    const badge = inferBadge(data.description || '', data.title || name);
    const ph = imgWrap.querySelector('.obj-image-placeholder');
    if (ph) ph.textContent = PLACEHOLDERS[badge] || '🔭';

    // Prefer the curated, hand-picked photo for featured objects (nicer,
    // higher-res) — but if that link ever breaks, fall back through the
    // live backend images (NASA → Wikipedia → Google) automatically.
    const primary = KNOWN_IMAGES[name] || data.image;
    if (primary) {
      const fallbacks = [data.image, ...(data.images || [])];
      showImage(primary, fallbacks);
    }

    badgeEl.textContent = badge;
    titleEl.textContent = data.title || name;
    extractEl.textContent = data.extract || 'No description available.';

    linksEl.innerHTML = '';
    if (data.wiki_link) {
      linksEl.innerHTML += `<a href="${data.wiki_link}" target="_blank" rel="noopener" class="obj-link obj-link-wiki">📖 Wikipedia</a>`;
    }
    linksEl.innerHTML += `<a href="https://www.nasa.gov/search/?q=${encodeURIComponent(name)}" target="_blank" rel="noopener" class="obj-link obj-link-nasa">🚀 NASA</a>`;
    linksEl.innerHTML += `<a href="https://exoplanetarchive.ipac.caltech.edu/index.html" target="_blank" rel="noopener" class="obj-link">🔬 Exoplanet Archive</a>`;

    loading.hidden = true;
    content.hidden = false;
  }

  async function loadObject(name) {
    const mySeq = ++detailsSeq;
    card.hidden = false;

    if (KNOWN_IMAGES[name]) {
      // Show a curated placeholder image + name instantly while the live
      // backend lookup (real title/description/best-available image) loads
      // in the background and then takes over via renderObject().
      content.hidden = false;
      loading.hidden = true;
      showImage(KNOWN_IMAGES[name]);
      badgeEl.textContent = inferBadge('', name);
      titleEl.textContent = name;
      extractEl.textContent = 'Loading…';
      linksEl.innerHTML = `<a href="https://en.wikipedia.org/wiki/${encodeURIComponent(name)}" target="_blank" rel="noopener" class="obj-link obj-link-wiki">📖 Wikipedia</a>`;
    } else {
      loading.hidden = false;
      content.hidden = true;
      card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    if (_detailCache.has(name)) { renderObject(name, _detailCache.get(name)); return; }

    try {
      const d = await objectDetails(name);
      if (mySeq !== detailsSeq) return; // a newer selection superseded this one

      // Backend already prioritizes NASA Images API, then Google Custom
      // Search (if configured), then Wikipedia as a last resort.
      const data = {
        title: d.title || name,
        description: d.description || '',
        extract: d.extract || '',
        image: d.image || KNOWN_IMAGES[name] || '',
        images: d.images || [],
        wiki_link: d.wiki_link || '',
      };

      _detailCache.set(name, data);
      renderObject(name, data);
    } catch(err) {
      if (mySeq !== detailsSeq) return;
      if (!KNOWN_IMAGES[name]) {
        loading.hidden = true;
        content.hidden = false;
        titleEl.textContent = name;
        extractEl.textContent = 'Could not load. Try another object or check your connection.';
        badgeEl.textContent = 'Unknown';
        linksEl.innerHTML = `<a href="https://en.wikipedia.org/wiki/${encodeURIComponent(name)}" target="_blank" rel="noopener" class="obj-link obj-link-wiki">📖 Wikipedia</a>`;
      }
    }
  }

  input.addEventListener('input', () => {
    const q = input.value.trim();
    clearBtn.hidden = q.length === 0;
    clearTimeout(debounceTimer);
    if (q.length < 2) { dropdown.hidden = true; return; }

    if (_searchCache.has(q)) { renderDropdown(_searchCache.get(q)); return; }

    const mySeq = ++searchSeq;
    debounceTimer = setTimeout(async () => {
      try {
        const results = await wikiSearch(q);
        if (mySeq !== searchSeq) return; // a newer query superseded this one
        _searchCache.set(q, results);
        renderDropdown(results);
      } catch(e) { if (mySeq === searchSeq) dropdown.hidden = true; }
    }, 280);
  });

  function renderDropdown(results) {
    dropdown.innerHTML = '';
    if (!results.length) { dropdown.hidden = true; return; }
    results.forEach(r => {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${r.title}</strong><span>${r.description}</span>`;
      li.addEventListener('click', () => {
        input.value = r.title;
        dropdown.hidden = true;
        clearBtn.hidden = false;
        setActiveChip(null);
        loadObject(r.title);
      });
      dropdown.appendChild(li);
    });
    dropdown.hidden = false;
  }

  document.addEventListener('click', e => {
    if (!e.target.closest('.explore-search-wrap')) dropdown.hidden = true;
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const q = input.value.trim();
      if (q) { dropdown.hidden = true; setActiveChip(null); loadObject(q); }
    }
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    clearBtn.hidden = true;
    dropdown.hidden = true;
    setActiveChip(null);
  });

  function setActiveChip(el) {
    document.querySelectorAll('.feat-chip').forEach(c => c.classList.remove('active'));
    if (el) el.classList.add('active');
  }

  document.querySelectorAll('.feat-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      input.value = chip.dataset.obj;
      clearBtn.hidden = false;
      dropdown.hidden = true;
      setActiveChip(chip);
      loadObject(chip.dataset.obj);
    });
  });

  const defaultObj = 'Orion Nebula';
  const defaultChip = document.querySelector(`.feat-chip[data-obj="${defaultObj}"]`);
  if (defaultChip) { setActiveChip(defaultChip); input.value = defaultObj; clearBtn.hidden = false; }
  loadObject(defaultObj);
})();

/* ── 7. Preset examples ────────────────────────────────────────────────── */
(function () {
  const PRESETS = {
    confirmed: {
      koi_model_snr: '39.3', koi_prad: '2.38', koi_period: '289.86',
      koi_depth: '492', koi_steff: '5518', koi_slogg: '4.44',
      koi_impact: '0.22', koi_duration: '7.41', koi_teq: '262',
      koi_max_mult_ev: '39.3', koi_max_sngle_ev: '7.9', koi_num_transits: '3',
      koi_srad: '0.979', koi_smass: '0.97', koi_smet: '-0.29',
    },
    candidate: {
      koi_model_snr: '18.2', koi_prad: '1.80', koi_period: '13.48',
      koi_depth: '350', koi_steff: '5200', koi_slogg: '4.50',
      koi_impact: '0.45', koi_duration: '3.12', koi_teq: '590',
      koi_max_mult_ev: '18.2', koi_max_sngle_ev: '5.4', koi_num_transits: '22',
      koi_srad: '0.88', koi_smass: '0.85', koi_smet: '0.04',
    },
    fp: {
      koi_model_snr: '505.6', koi_prad: '33.46', koi_period: '1.74',
      koi_depth: '8079', koi_steff: '5805', koi_slogg: '4.56',
      koi_impact: '1.28', koi_duration: '2.41', koi_teq: '1395',
      koi_max_mult_ev: '541.9', koi_max_sngle_ev: '39.1', koi_num_transits: '621',
      koi_srad: '0.836', koi_smass: '0.961', koi_smet: '-0.52',
    },
  };

  document.querySelectorAll('.preset-card').forEach(card => {
    card.addEventListener('click', () => {
      const values = PRESETS[card.dataset.preset];
      if (!values) return;
      document.querySelectorAll('#predict-form input[name]').forEach(inp => {
        if (values[inp.name] !== undefined) {
          inp.value = values[inp.name];
          inp.classList.add('prefilled');
        } else { inp.value = ''; inp.classList.remove('prefilled'); }
      });
      const adv = document.getElementById('advanced-fields');
      const lbl = document.getElementById('advanced-label');
      if (adv && adv.hidden) { adv.hidden = false; if (lbl) lbl.textContent = '－ Hide advanced parameters'; }
      document.getElementById('predict-form').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
})();

/* ── 8. Advanced toggle ────────────────────────────────────────────────── */
(function () {
  const toggle = document.getElementById('advanced-toggle');
  const fields = document.getElementById('advanced-fields');
  const label  = document.getElementById('advanced-label');
  if (!toggle || !fields) return;
  toggle.addEventListener('click', () => {
    fields.hidden = !fields.hidden;
    label.textContent = fields.hidden ? '＋ Show advanced parameters' : '－ Hide advanced parameters';
  });
})();

/* ── 9. Clear button ───────────────────────────────────────────────────── */
(function () {
  document.getElementById('clear-btn')?.addEventListener('click', () => {
    document.querySelectorAll('#predict-form input[name]').forEach(inp => { inp.value = ''; inp.classList.remove('prefilled'); });
    const idle = document.querySelector('.result-idle');
    const output = document.querySelector('.result-output');
    if (idle) idle.hidden = false;
    if (output) output.hidden = true;
  });
})();

/* ── 10. Prediction form (client-side Gaussian Naive Bayes) ──────────── */
(function () {
  const form = document.getElementById('predict-form');
  if (!form) return;

  const idle      = document.querySelector('.result-idle');
  const output    = document.querySelector('.result-output');
  const labelEl   = document.getElementById('result-label');
  const confEl    = document.getElementById('result-confidence');
  const probsEl   = document.getElementById('result-probs');
  const missingEl = document.getElementById('result-missing');
  const btn       = form.querySelector('.submit-btn');
  const btnText   = btn.querySelector('.btn-text');
  const btnSpinner = btn.querySelector('.btn-spinner');

  const CLASS_COLORS = { 'CONFIRMED': '#10b981', 'CANDIDATE': '#f59e0b', 'FALSE POSITIVE': '#ef4444' };

  /* ── Predict via the real trained backend model (RandomForest         */
  /* pipeline in celesta/model.joblib, retrained on the full KOI         */
  /* dataset) — no client-side approximation, so results match the      */
  /* dashboard's reported performance numbers.                          */
  async function predict(data) {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Prediction request failed');
    return await res.json();
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const data = {};
    form.querySelectorAll('input[name]').forEach(inp => {
      const v = inp.value.trim();
      if (v !== '') data[inp.name] = v;
    });

    if (Object.keys(data).length === 0) {
      missingEl.textContent = 'Enter at least one field to classify.';
      missingEl.hidden = false;
      return;
    }

    btn.disabled = true;
    btnText.hidden = true;
    btnSpinner.hidden = false;

    let result = null;
    try {
      result = await predict(data);
    } catch (err) {
      result = null;
    }

    btn.disabled = false;
    btnText.hidden = false;
    btnSpinner.hidden = true;

    if (!result || result.error) {
      output.hidden = true;
      idle.hidden = false;
      missingEl.textContent = 'Could not reach the classifier. Please try again.';
      missingEl.hidden = false;
      return;
    }

    idle.hidden = true;
    output.hidden = false;

    const pred = result.prediction;
    const cls = pred === 'CONFIRMED' ? 'confirmed' : pred === 'CANDIDATE' ? 'candidate' : 'fp';

    labelEl.textContent = pred;
    labelEl.className = `result-label ${cls}`;
    confEl.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

    probsEl.innerHTML = '';
    Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).forEach(([c, prob]) => {
      const color = CLASS_COLORS[c] || '#6366f1';
      probsEl.innerHTML += `
        <div class="prob-row">
          <span>${c}</span>
          <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:${prob * 100}%;background:${color}"></div>
          </div>
          <span class="prob-val" style="color:${color}">${(prob * 100).toFixed(1)}%</span>
        </div>`;
    });

    if (result.missing_fields.length > 0) {
      missingEl.textContent = `${result.missing_fields.length} missing field(s) auto-filled with dataset medians.`;
      missingEl.hidden = false;
    } else {
      missingEl.hidden = true;
    }
  });
})();
