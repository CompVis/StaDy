document.querySelector('#copy-citation').addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#citation').textContent);
    status.textContent = 'Citation copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#citation'));
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});

const relatedWork = document.querySelector('.related-work-flyout');
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && relatedWork.open) {
    relatedWork.open = false;
    relatedWork.querySelector('summary').focus();
  }
});
document.addEventListener('click', event => {
  if (!relatedWork.contains(event.target)) relatedWork.open = false;
});
