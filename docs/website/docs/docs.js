(() => {
  const root = document.documentElement;
  const body = document.body;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(pointer: fine)").matches;
  const details = document.querySelector(".mobile-nav details");
  const path = window.location.pathname;

  const section = path.includes("/getting-started/")
    ? "getting-started"
    : path.includes("/concepts/")
      ? "concepts"
      : path.includes("/guides/")
        ? "guides"
        : path.includes("/reference/")
          ? "reference"
          : "home";

  body.dataset.docSection = section;

  if (window.hljs) window.hljs.highlightAll();


  document.querySelectorAll(".mobile-nav a").forEach((link) => {
    link.addEventListener("click", () => {
      if (details) details.open = false;
    });
  });

  const revealTargets = [
    ...document.querySelectorAll(
      ".content > h2, .content > p, .content > .note, .content > .table-wrap, .content > pre, .content > .page-nav, .content > section:not(.section-intro)"
    ),
  ];

  if (reducedMotion || !("IntersectionObserver" in window)) {
    revealTargets.forEach((node) => node.classList.add("is-visible"));
  } else {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.1, rootMargin: "0px 0px -7% 0px" });

    revealTargets.forEach((node) => observer.observe(node));
  }

  /* Accent theme rotates every THEME_STEP px of scroll. */
  const THEME_STEP = 600;
  const THEME_THEMES = [
    ["#9fe870", "159,232,112", "#71d6ff", "113,214,255"],
    ["#71d6ff", "113,214,255", "#b392ff", "179,146,255"],
    ["#b392ff", "179,146,255", "#ff7aa8", "255,122,168"],
    ["#ff7aa8", "255,122,168", "#ffbd66", "255,189,102"],
    ["#ffbd66", "255,189,102", "#9fe870", "159,232,112"],
  ];
  let themeStep = -1;

  let scrollFrame = 0;
  const updateScroll = () => {
    scrollFrame = 0;
    const top = window.scrollY || document.documentElement.scrollTop;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = max > 0 ? Math.min(1, Math.max(0, top / max)) : 0;
    root.style.setProperty("--scroll", ratio.toFixed(4));
    const step = Math.floor(top / THEME_STEP) % THEME_THEMES.length;
    if (step !== themeStep) {
      themeStep = step;
      const [a, ar, b, br] = THEME_THEMES[step];
      [["--accent", a], ["--accent-rgb", ar], ["--accent-2", b], ["--accent-2-rgb", br]]
        .forEach(([k, v]) => { body.style.setProperty(k, v); root.style.setProperty(k, v); });
    }
    if (!reducedMotion) {
      root.style.setProperty("--hero-shift", `${Math.min(22, top * 0.028)}px`);
    }
  };

  const requestScroll = () => {
    if (scrollFrame) return;
    scrollFrame = window.requestAnimationFrame(updateScroll);
  };

  updateScroll();
  window.addEventListener("scroll", requestScroll, { passive: true });
  window.addEventListener("resize", requestScroll, { passive: true });

  if (!reducedMotion && finePointer) {
    let pointerFrame = 0;

    window.addEventListener("pointermove", (event) => {
      if (pointerFrame) return;
      pointerFrame = window.requestAnimationFrame(() => {
        root.style.setProperty("--mx", `${event.clientX}px`);
        root.style.setProperty("--my", `${event.clientY}px`);
        pointerFrame = 0;
      });
    }, { passive: true });

    document.querySelectorAll(".card").forEach((card) => {
      card.addEventListener("pointermove", (event) => {
        const rect = card.getBoundingClientRect();
        const px = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
        const py = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
        card.style.setProperty("--spot-x", `${(px * 100).toFixed(1)}%`);
        card.style.setProperty("--spot-y", `${(py * 100).toFixed(1)}%`);
        card.style.setProperty("--tilt-x", `${((0.5 - py) * 2.8).toFixed(2)}deg`);
        card.style.setProperty("--tilt-y", `${((px - 0.5) * 3.6).toFixed(2)}deg`);
      });

      card.addEventListener("pointerleave", () => {
        card.style.setProperty("--spot-x", "50%");
        card.style.setProperty("--spot-y", "50%");
        card.style.setProperty("--tilt-x", "0deg");
        card.style.setProperty("--tilt-y", "0deg");
      });
    });
  }
})();