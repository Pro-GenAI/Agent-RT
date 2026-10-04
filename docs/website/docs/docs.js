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

  let scrollFrame = 0;
  const updateScroll = () => {
    scrollFrame = 0;
    const top = window.scrollY || document.documentElement.scrollTop;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = max > 0 ? Math.min(1, Math.max(0, top / max)) : 0;
    root.style.setProperty("--scroll", ratio.toFixed(4));
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