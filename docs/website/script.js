(() => {
  const root = document.documentElement;
  root.classList.add("js");

  if (!root.dataset.motion) {
    root.dataset.motion = "full";
  }
  const reducedMotion = root.dataset.motion === "reduce";
  const finePointer = window.matchMedia("(pointer: fine)").matches;
  const hasIO = "IntersectionObserver" in window;
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const installCommands = {
    python: "pip install agent-rt",
    typescript: "npm install agent-rt",
  };

  const header = document.querySelector(".site-header");
  const progressBar = document.getElementById("page-progress-bar");
  const navToggle = document.querySelector(".nav-toggle");
  const siteNav = document.getElementById("site-nav");
  const toast = document.getElementById("copy-toast");

  /* ---------------------------------------------------------------
     Motion toggle: animations are on by default; the visitor's
     choice is remembered for this browser only.
     --------------------------------------------------------------- */
  document.querySelectorAll("[data-motion-toggle]").forEach((button) => {
    button.setAttribute("aria-pressed", String(!reducedMotion));
    button.setAttribute("aria-label", reducedMotion ? "Turn animations on" : "Turn animations off");
    button.dataset.tooltip = reducedMotion ? "Animations off — click to turn on" : "Animations on — click to turn off";
    button.addEventListener("click", () => {
      try {
        window.localStorage.setItem("agent-rt-motion", reducedMotion ? "full" : "reduce");
      } catch {
        // Storage can be unavailable (private mode); the reload still applies nothing persistent.
      }
      window.location.reload();
    });
  });

  /* ---------------------------------------------------------------
     Split headline into animated characters
     --------------------------------------------------------------- */
  const splitTargets = document.querySelectorAll("[data-split]");
  const gradientChars = [];

  const heading = document.querySelector(".hero-title");
  if (heading && splitTargets.length) {
    heading.setAttribute("aria-label", [...splitTargets].map((target) => target.textContent.trim()).join(" "));
  }

  splitTargets.forEach((target, lineIndex) => {
    const text = target.textContent.trim();
    const visual = document.createElement("span");
    visual.setAttribute("aria-hidden", "true");
    visual.style.setProperty("--line-delay", `${lineIndex * 260}ms`);

    let charIndex = 0;
    const words = text.split(" ");
    words.forEach((word, wordIndex) => {
      const wrap = document.createElement("span");
      wrap.className = "char-wrap";
      [...word].forEach((character) => {
        const char = document.createElement("span");
        char.className = "char";
        char.textContent = character;
        char.style.setProperty("--i", String(charIndex));
        charIndex += 1;
        wrap.appendChild(char);
        if (target.closest(".title-gradient")) gradientChars.push(char);
      });
      visual.appendChild(wrap);
      if (wordIndex < words.length - 1) visual.appendChild(document.createTextNode(" "));
    });

    target.replaceChildren(visual);
    target.classList.add("is-split");
  });

  const layoutGradientChars = () => {
    gradientChars.forEach((char) => {
      char.style.setProperty("--x", `${char.offsetLeft + char.parentElement.offsetLeft}px`);
    });
  };
  layoutGradientChars();
  window.addEventListener("resize", layoutGradientChars, { passive: true });
  if (document.fonts?.ready) document.fonts.ready.then(layoutGradientChars);

  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => document.querySelector(".hero")?.classList.add("is-ready"));
  });

  /* ---------------------------------------------------------------
     Scroll state: header, progress, architecture assembly, words
     --------------------------------------------------------------- */
  const archStage = document.getElementById("arch-stage");
  const scrollWords = document.querySelector("[data-scroll-words]");
  let wordNodes = [];

  if (scrollWords) {
    const highlight = new Set(["explicit", "contracts", "hidden"]);
    const words = scrollWords.textContent.trim().split(/\s+/);
    scrollWords.setAttribute("aria-label", words.join(" "));
    scrollWords.textContent = "";
    words.forEach((word, index) => {
      const span = document.createElement("span");
      span.className = "w";
      span.setAttribute("aria-hidden", "true");
      if (highlight.has(word.toLowerCase().replace(/[^a-z]/g, ""))) span.classList.add("hl");
      span.textContent = word;
      scrollWords.appendChild(span);
      if (index < words.length - 1) scrollWords.appendChild(document.createTextNode(" "));
    });
    wordNodes = [...scrollWords.querySelectorAll(".w")];
    if (reducedMotion) wordNodes.forEach((node) => node.classList.add("lit"));
  }

  /* Accent theme (logo, nav, scrollbar, highlights) rotates every THEME_STEP px of scroll. */
  const THEME_STEP = 600;
  const THEME_COLORS = ["#c8ff4d", "#5ee0f1", "#6a9fff", "#a98bff", "#ff7ab8", "#ffc76a"];
  let themeStep = -1;

  let scrollFrame = 0;
  const updateScroll = () => {
    scrollFrame = 0;
    const scrollTop = window.scrollY || root.scrollTop;
    const viewport = window.innerHeight;
    const max = root.scrollHeight - viewport;
    const ratio = max > 0 ? clamp(scrollTop / max, 0, 1) : 0;

    header?.classList.toggle("scrolled", scrollTop > 24);
    const step = Math.floor(scrollTop / THEME_STEP) % THEME_COLORS.length;
    if (step !== themeStep) {
      themeStep = step;
      root.style.setProperty("--acid", THEME_COLORS[step]);
    }
    if (progressBar) progressBar.style.width = `${ratio * 100}%`;

    if (reducedMotion) return;

    if (archStage) {
      const rect = archStage.getBoundingClientRect();
      const progress = clamp((viewport * 0.95 - rect.top) / (viewport * 0.6), 0, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      archStage.style.setProperty("--p", eased.toFixed(4));
    }

    if (wordNodes.length) {
      const rect = scrollWords.getBoundingClientRect();
      const progress = clamp((viewport * 0.82 - rect.top) / (rect.height + viewport * 0.3), 0, 1);
      const lit = Math.round(progress * wordNodes.length);
      wordNodes.forEach((node, index) => node.classList.toggle("lit", index < lit));
    }
  };

  const requestScroll = () => {
    if (!scrollFrame) scrollFrame = window.requestAnimationFrame(updateScroll);
  };
  updateScroll();
  window.addEventListener("scroll", requestScroll, { passive: true });
  window.addEventListener("resize", requestScroll, { passive: true });

  /* ---------------------------------------------------------------
     Navigation
     --------------------------------------------------------------- */
  const setNavOpen = (open) => {
    navToggle?.setAttribute("aria-expanded", String(open));
    siteNav?.classList.toggle("open", open);
    document.body.classList.toggle("nav-open", open);
  };

  navToggle?.addEventListener("click", () => {
    setNavOpen(navToggle.getAttribute("aria-expanded") !== "true");
  });
  siteNav?.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => setNavOpen(false)));
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || navToggle?.getAttribute("aria-expanded") !== "true") return;
    setNavOpen(false);
    navToggle.focus();
  });
  window.matchMedia("(min-width: 861px)").addEventListener("change", (event) => {
    if (event.matches) setNavOpen(false);
  });

  const navLinks = [...(siteNav?.querySelectorAll('a[href^="#"]') || [])];
  const observedSections = navLinks
    .map((link) => {
      const section = document.getElementById(link.getAttribute("href").slice(1));
      return section ? { link, section } : null;
    })
    .filter(Boolean);

  if (hasIO && observedSections.length) {
    const sectionObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        observedSections.forEach(({ link, section }) => {
          const active = section === entry.target;
          link.classList.toggle("active", active);
          if (active) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    observedSections.forEach(({ section }) => sectionObserver.observe(section));
  }

  /* ---------------------------------------------------------------
     Tabs: install command + quickstart code
     --------------------------------------------------------------- */
  const installTabs = document.querySelectorAll(".install-tab");
  installTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const command = installCommands[tab.dataset.install];
      if (!command) return;
      installTabs.forEach((item) => {
        const selected = item === tab;
        item.classList.toggle("active", selected);
        item.setAttribute("aria-selected", String(selected));
      });
      const commandNode = document.getElementById("hero-install-command");
      if (commandNode) {
        commandNode.classList.remove("command-swap");
        void commandNode.offsetWidth;
        commandNode.textContent = command;
        commandNode.classList.add("command-swap");
      }
    });
  });

  const codeTabs = [...document.querySelectorAll(".code-tab")];
  const codePanels = document.querySelectorAll(".code-panel");
  const codeCopyButton = document.querySelector(".code-copy");

  const selectCodeTab = (tab, focus = false) => {
    const language = tab.dataset.codeTab;
    codeTabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    codePanels.forEach((panel) => {
      const selected = panel.id === `${language}-panel`;
      panel.classList.toggle("active", selected);
      panel.hidden = !selected;
      if (selected && !reducedMotion) {
        panel.classList.remove("panel-enter");
        void panel.offsetWidth;
        panel.classList.add("panel-enter");
      }
    });
    if (codeCopyButton) codeCopyButton.dataset.copyTarget = `${language}-code`;
    if (focus) tab.focus();
  };

  codeTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectCodeTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      selectCodeTab(codeTabs[(index + step + codeTabs.length) % codeTabs.length], true);
    });
  });

  /* ---------------------------------------------------------------
     Copy to clipboard
     --------------------------------------------------------------- */
  let toastTimer;
  const showToast = (message) => {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("show"), 1700);
  };

  const copyText = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    if (!copied) throw new Error("Copy command was not available.");
  };

  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget || "");
      const text = target?.textContent?.trim();
      if (!text) return;
      try {
        await copyText(text);
        button.classList.remove("copy-success");
        void button.offsetWidth;
        button.classList.add("copy-success");
        const label = button.querySelector(".copy-label");
        if (label) {
          label.textContent = "Copied";
          window.setTimeout(() => { label.textContent = "Copy"; button.classList.remove("copy-success"); }, 1400);
        }
        showToast("Copied to clipboard");
      } catch {
        showToast("Select and copy manually");
      }
    });
  });

  /* ---------------------------------------------------------------
     Reveal on scroll
     --------------------------------------------------------------- */
  const staggerGroup = (selector, step) => {
    document.querySelectorAll(selector).forEach((group) => {
      [...group.children].filter((node) => node.classList.contains("reveal")).forEach((node, index) => {
        node.style.setProperty("--reveal-delay", `${Math.min(index * step, 480)}ms`);
      });
    });
  };
  staggerGroup(".hero-copy", 110);
  staggerGroup(".principles-grid", 110);
  staggerGroup(".bento-grid", 80);
  staggerGroup(".benchmark-grid", 110);
  staggerGroup(".section-heading", 90);
  staggerGroup(".section-heading > div", 90);
  staggerGroup(".quickstart-copy", 80);

  const heroVisual = document.querySelector(".hero-visual");
  heroVisual?.style.setProperty("--reveal-delay", "520ms");

  const revealNodes = document.querySelectorAll(".reveal");
  if (reducedMotion || !hasIO) {
    revealNodes.forEach((node) => node.classList.add("visible"));
  } else {
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    revealNodes.forEach((node) => revealObserver.observe(node));
  }

  /* ---------------------------------------------------------------
     Benchmarks: count-up numbers and bars
     --------------------------------------------------------------- */
  const animateNumber = (element, duration = 1300) => {
    const textNode = [...element.childNodes].find((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim());
    if (!textNode) return;
    const target = Number.parseFloat(textNode.textContent);
    if (!Number.isFinite(target)) return;
    const decimals = (textNode.textContent.trim().split(".")[1] || "").length;
    const start = performance.now();
    const frame = (now) => {
      const progress = clamp((now - start) / duration, 0, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      textNode.nodeValue = (target * eased).toFixed(decimals);
      if (progress < 1) window.requestAnimationFrame(frame);
    };
    textNode.nodeValue = (0).toFixed(decimals);
    window.requestAnimationFrame(frame);
  };

  const benchmarkSection = document.getElementById("benchmarks");
  const benchmarkTrigger = benchmarkSection?.querySelector(".benchmark-grid");
  if (benchmarkSection && benchmarkTrigger && !reducedMotion && hasIO) {
    const benchmarkObserver = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      benchmarkObserver.disconnect();
      benchmarkSection.classList.add("metrics-live");
      benchmarkSection.querySelectorAll(".benchmark-value, .dual-stat strong").forEach((node, index) => {
        window.setTimeout(() => animateNumber(node), 150 + index * 120);
      });
    }, { threshold: 0.3 });
    benchmarkObserver.observe(benchmarkTrigger);
  } else {
    benchmarkSection?.classList.add("metrics-live");
  }

  /* ---------------------------------------------------------------
     Hero console: illustrative run trace typed in a loop
     --------------------------------------------------------------- */
  const consoleNode = document.getElementById("hero-console");
  const traceList = document.getElementById("hero-trace");
  const meterTurns = document.getElementById("meter-turns");
  const statTokens = document.getElementById("stat-tokens");
  const statSpans = document.getElementById("stat-spans");

  const scenarios = [
    [
      ["01", "ev-model", "text_delta", '"Checking the SLA first…"', "12ms", ""],
      ["02", "ev-tool", "tool_call_delta", 'search_docs(query="SLA")', "schema ✓", ""],
      ["03", "ev-policy", "policy.check", "side_effect=read", "allow", "ok"],
      ["04", "ev-tool", "tool_completed", "search_docs → 3 results", "38ms", ""],
      ["05", "ev-state", "checkpoint", "turn_02 committed", "saved", ""],
      ["06", "ev-policy", "approval.request", "send_email → human", "paused", "warn"],
      ["✓", "ev-done", "termination_reason", "waiting_for_approval", "", "", true],
    ],
    [
      ["01", "ev-state", "checkpoint.load", "run_42 · turn_02", "restored", ""],
      ["02", "ev-policy", "approval.resolve", "send_email", "approved", "ok"],
      ["03", "ev-policy", "policy.check", "side_effect=write", "allow", "ok"],
      ["04", "ev-tool", "tool_completed", "send_email → queued", "41ms", ""],
      ["05", "ev-model", "text_delta", '"Done — summary sent."', "9ms", ""],
      ["06", "ev-state", "checkpoint", "turn_03 committed", "saved", ""],
      ["✓", "ev-done", "termination_reason", "completed", "", "", true],
    ],
  ];

  const buildLine = ([n, evClass, event, message, meta, metaClass, end]) => {
    const li = document.createElement("li");
    li.className = end ? "tl tl-end" : "tl";
    li.innerHTML = `<span class="tl-n"></span><span class="tl-ev ${evClass}"></span><span class="tl-msg"></span><span class="tl-meta ${metaClass}"></span>`;
    li.children[0].textContent = n;
    li.children[1].textContent = event;
    li.children[3].textContent = meta;
    li.dataset.msg = message;
    return li;
  };

  let consoleVisible = true;
  const waitUntilVisible = async () => {
    while (!consoleVisible || document.hidden) await wait(400);
  };

  const typeInto = async (node, text) => {
    const caret = document.createElement("span");
    caret.className = "caret";
    node.textContent = "";
    node.appendChild(caret);
    for (let i = 1; i <= text.length; i += 1) {
      node.textContent = text.slice(0, i);
      node.appendChild(caret);
      await wait(14 + Math.random() * 22);
    }
    await wait(120);
    caret.remove();
  };

  const runConsole = async () => {
    if (!consoleNode || !traceList) return;
    consoleNode.classList.add("is-animating");
    let scenarioIndex = 0;
    let tokens = 0;

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const lines = scenarios[scenarioIndex].map(buildLine);
      traceList.replaceChildren(...lines);
      if (meterTurns) meterTurns.style.width = scenarioIndex === 0 ? "12%" : "50%";
      await wait(500);

      for (let index = 0; index < lines.length; index += 1) {
        await waitUntilVisible();
        const line = lines[index];
        lines.forEach((item) => item.classList.remove("fresh"));
        line.classList.add("on", "fresh");
        await typeInto(line.querySelector(".tl-msg"), line.dataset.msg);

        tokens += 90 + Math.round(Math.random() * 160);
        if (statTokens) statTokens.textContent = tokens.toLocaleString("en-US");
        if (statSpans) statSpans.textContent = String(index + 1 + scenarioIndex * 6);
        if (meterTurns) meterTurns.style.width = `${clamp((scenarioIndex * 6 + index + 2) / 14, 0, 1) * 100}%`;
        await wait(380 + Math.random() * 260);
      }

      lines.forEach((item) => item.classList.remove("fresh"));
      await wait(3400);
      await waitUntilVisible();
      lines.forEach((item) => item.classList.remove("on"));
      await wait(500);
      scenarioIndex = (scenarioIndex + 1) % scenarios.length;
      if (scenarioIndex === 0) tokens = 0;
    }
  };

  if (consoleNode && !reducedMotion) {
    if (hasIO) {
      new IntersectionObserver((entries) => {
        consoleVisible = entries[0].isIntersecting;
      }).observe(consoleNode);
    }
    window.setTimeout(runConsole, 900);
  }

  /* ---------------------------------------------------------------
     Neural field canvas
     --------------------------------------------------------------- */
  const canvas = document.getElementById("neural-field");
  if (canvas && canvas.getContext) {
    const ctx = canvas.getContext("2d");
    const palette = ["200,255,77", "94,224,241", "106,159,255", "169,139,255"];
    let width = 0;
    let height = 0;
    let dpr = 1;
    let nodes = [];
    let packets = [];
    let running = false;
    let rafId = 0;
    const pointer = { x: -9999, y: -9999, active: false };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.round(clamp((width * height) / 16000, 28, 96));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.28,
        vy: (Math.random() - 0.5) * 0.28,
        r: Math.random() * 1.6 + 0.6,
        hub: Math.random() < 0.08,
        color: palette[Math.floor(Math.random() * palette.length)],
      }));
      packets = [];
    };

    const linkDistance = () => clamp(width / 9, 110, 170);

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      const maxDist = linkDistance();
      const maxDistSq = maxDist * maxDist;

      for (const node of nodes) {
        node.x += node.vx;
        node.y += node.vy;
        if (node.x < -20) node.x = width + 20;
        if (node.x > width + 20) node.x = -20;
        if (node.y < -20) node.y = height + 20;
        if (node.y > height + 20) node.y = -20;

        if (pointer.active) {
          const dx = pointer.x - node.x;
          const dy = pointer.y - node.y;
          const distSq = dx * dx + dy * dy;
          if (distSq < 40000 && distSq > 1) {
            const force = (1 - distSq / 40000) * 0.012;
            node.x += dx * force;
            node.y += dy * force;
          }
        }
      }

      ctx.lineWidth = 1;
      for (let i = 0; i < nodes.length; i += 1) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j += 1) {
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const distSq = dx * dx + dy * dy;
          if (distSq > maxDistSq) continue;
          const alpha = (1 - Math.sqrt(distSq) / maxDist) * 0.22;
          ctx.strokeStyle = `rgba(130,160,255,${alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();

          if (packets.length < 26 && Math.random() < 0.0009) {
            packets.push({ a, b, t: 0, speed: 0.008 + Math.random() * 0.014, color: Math.random() < 0.5 ? a.color : b.color });
          }
        }
      }

      if (pointer.active) {
        for (const node of nodes) {
          const dx = pointer.x - node.x;
          const dy = pointer.y - node.y;
          const distSq = dx * dx + dy * dy;
          if (distSq > 32000) continue;
          const alpha = (1 - distSq / 32000) * 0.45;
          ctx.strokeStyle = `rgba(200,255,77,${alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.moveTo(pointer.x, pointer.y);
          ctx.lineTo(node.x, node.y);
          ctx.stroke();
        }
      }

      for (const node of nodes) {
        ctx.fillStyle = `rgba(${node.color},${node.hub ? 0.95 : 0.6})`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.hub ? node.r + 1.4 : node.r, 0, Math.PI * 2);
        ctx.fill();
        if (node.hub) {
          ctx.fillStyle = `rgba(${node.color},0.12)`;
          ctx.beginPath();
          ctx.arc(node.x, node.y, 9, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      packets = packets.filter((packet) => {
        packet.t += packet.speed;
        if (packet.t >= 1) return false;
        const x = packet.a.x + (packet.b.x - packet.a.x) * packet.t;
        const y = packet.a.y + (packet.b.y - packet.a.y) * packet.t;
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
        gradient.addColorStop(0, `rgba(${packet.color},0.95)`);
        gradient.addColorStop(1, `rgba(${packet.color},0)`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
        return true;
      });
    };

    const loop = () => {
      draw();
      rafId = running ? window.requestAnimationFrame(loop) : 0;
    };

    const setRunning = (next) => {
      if (reducedMotion) return;
      if (next === running) return;
      running = next;
      if (running && !rafId) rafId = window.requestAnimationFrame(loop);
    };

    resize();
    draw();

    let resizeTimer;
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => { resize(); if (!running) draw(); }, 150);
    }, { passive: true });

    if (!reducedMotion) {
      const hero = document.querySelector(".hero");
      if (hasIO && hero) {
        new IntersectionObserver((entries) => setRunning(entries[0].isIntersecting && !document.hidden)).observe(hero);
      } else {
        setRunning(true);
      }
      document.addEventListener("visibilitychange", () => {
        if (document.hidden) setRunning(false);
        else if (hero && hero.getBoundingClientRect().bottom > 0) setRunning(true);
      });

      if (finePointer && hero) {
        hero.addEventListener("pointermove", (event) => {
          const rect = canvas.getBoundingClientRect();
          pointer.x = event.clientX - rect.left;
          pointer.y = event.clientY - rect.top;
          pointer.active = true;
        }, { passive: true });
        hero.addEventListener("pointerleave", () => { pointer.active = false; });
      }
    }
  }

  /* ---------------------------------------------------------------
     Pointer effects: glow, tilt, spotlight, magnetic buttons
     --------------------------------------------------------------- */
  if (!reducedMotion && finePointer) {
    root.classList.add("motion-pointer");

    let pointerFrame = 0;
    let lastEvent;
    window.addEventListener("pointermove", (event) => {
      lastEvent = event;
      if (pointerFrame) return;
      pointerFrame = window.requestAnimationFrame(() => {
        pointerFrame = 0;
        root.style.setProperty("--pointer-x", `${lastEvent.clientX}px`);
        root.style.setProperty("--pointer-y", `${lastEvent.clientY}px`);
      });
    }, { passive: true });

    if (heroVisual && consoleNode) {
      heroVisual.addEventListener("pointermove", (event) => {
        const rect = heroVisual.getBoundingClientRect();
        const px = clamp((event.clientX - rect.left) / rect.width, 0, 1);
        const py = clamp((event.clientY - rect.top) / rect.height, 0, 1);
        consoleNode.style.setProperty("--tilt-x", `${((0.5 - py) * 7).toFixed(2)}deg`);
        consoleNode.style.setProperty("--tilt-y", `${((px - 0.5) * 9).toFixed(2)}deg`);
        consoleNode.style.setProperty("--stage-x", `${px * 100}%`);
        consoleNode.style.setProperty("--stage-y", `${py * 100}%`);
      });
      heroVisual.addEventListener("pointerleave", () => {
        consoleNode.style.setProperty("--tilt-x", "0deg");
        consoleNode.style.setProperty("--tilt-y", "0deg");
      });
    }

    document.querySelectorAll(".feature-card, .benchmark-card, .comparison-panel").forEach((panel) => {
      panel.addEventListener("pointermove", (event) => {
        const rect = panel.getBoundingClientRect();
        panel.style.setProperty("--spot-x", `${event.clientX - rect.left}px`);
        panel.style.setProperty("--spot-y", `${event.clientY - rect.top}px`);
      });
    });

    document.querySelectorAll(".button").forEach((button) => {
      button.addEventListener("pointermove", (event) => {
        const rect = button.getBoundingClientRect();
        const x = clamp((event.clientX - rect.left) / rect.width, 0, 1) - 0.5;
        const y = clamp((event.clientY - rect.top) / rect.height, 0, 1) - 0.5;
        button.style.setProperty("--magnet-x", `${(x * 8).toFixed(2)}px`);
        button.style.setProperty("--magnet-y", `${(y * 6).toFixed(2)}px`);
      });
      button.addEventListener("pointerleave", () => {
        button.style.setProperty("--magnet-x", "0px");
        button.style.setProperty("--magnet-y", "0px");
      });
    });
  }
})();
