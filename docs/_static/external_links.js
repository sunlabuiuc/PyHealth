// Open every external link (http/https, different host) in a new tab.
document.addEventListener("DOMContentLoaded", () => {
  for (const a of document.querySelectorAll('a[href^="http"]')) {
    if (!a.href.includes(window.location.host)) {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    }
  }
});
