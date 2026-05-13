// Onboarding tour for new users
// Shows only once per user (stored in localStorage)

const ONBOARDING_KEY = 'processa_onboarding_done';
const ONBOARDING_VERSION = '1.0';

const consultorSteps = [
  {
    target: '#btn-new-process',
    title: '🚀 Crea tu primer proceso',
    text: 'Empieza describiendo un proceso de tu organización. Puedes usar texto libre o subir documentos.',
    position: 'bottom'
  },
  {
    target: '.sidebar-nav',
    title: '📊 Tu workspace',
    text: 'Aquí verás todos tus procesos y su estado de análisis.',
    position: 'right'
  },
  {
    target: '#notif-bell',
    title: '🔔 Notificaciones',
    text: 'Te avisaremos cuando tus colaboradores terminen su aporte o cuando un análisis esté listo.',
    position: 'bottom'
  },
  {
    target: '.user-info',
    title: '👤 Tu perfil',
    text: 'Como consultor puedes crear procesos, invitar colaboradores y ver los análisis completos.',
    position: 'top'
  }
];

const colaboradorSteps = [
  {
    target: '.collab-banner',
    title: '👋 Bienvenido a Processa AI',
    text: 'Has sido invitado a colaborar en uno o más procesos. Tu aporte ayudará a optimizarlos.',
    position: 'bottom'
  },
  {
    target: '.process-card',
    title: '📋 Tu proceso asignado',
    text: 'Abre el workspace y describe las actividades que realizas en este proceso.',
    position: 'right'
  },
  {
    target: '#notif-bell',
    title: '🔔 Notificaciones',
    text: 'Recibirás actualizaciones sobre los procesos en los que colaboras.',
    position: 'bottom'
  }
];

function createTooltip(step, currentStep, totalSteps) {
  const tooltip = document.createElement('div');
  tooltip.id = 'onboarding-tooltip';
  tooltip.innerHTML = `
    <div class="onboarding-overlay"></div>
    <div class="onboarding-box">
      <div class="onboarding-header">
        <span class="onboarding-step">${currentStep + 1}/${totalSteps}</span>
        <button class="onboarding-close" onclick="skipOnboarding()">✕</button>
      </div>
      <h3 class="onboarding-title">${step.title}</h3>
      <p class="onboarding-text">${step.text}</p>
      <div class="onboarding-footer">
        <button class="btn-onboarding-skip" onclick="skipOnboarding()">
          Saltar tour
        </button>
        <button class="btn-onboarding-next" onclick="nextOnboardingStep()">
          ${currentStep === totalSteps - 1 ? 'Empezar ✓' : 'Siguiente →'}
        </button>
      </div>
      <div class="onboarding-dots">
        ${Array.from({length: totalSteps}, (_, i) =>
          `<span class="dot ${i === currentStep ? 'active' : ''}"></span>`
        ).join('')}
      </div>
    </div>
  `;
  return tooltip;
}

let __onboardingSteps = [];
let __onboardingIndex = 0;

function startOnboarding(role) {
  const done = localStorage.getItem(ONBOARDING_KEY);
  if (done === ONBOARDING_VERSION) return;

  __onboardingSteps = (role || '').toLowerCase() === 'colaborador'
    ? colaboradorSteps
    : consultorSteps;
  __onboardingIndex = 0;

  showOnboardingStep();
}

function showOnboardingStep() {
  const existing = document.getElementById('onboarding-tooltip');
  if (existing) existing.remove();

  if (__onboardingIndex >= __onboardingSteps.length) {
    finishOnboarding();
    return;
  }

  const step = __onboardingSteps[__onboardingIndex];
  const target = document.querySelector(step.target);

  // If target not found, skip to next
  if (!target) {
    __onboardingIndex++;
    showOnboardingStep();
    return;
  }

  target.classList.add('onboarding-highlight');

  const tooltip = createTooltip(step, __onboardingIndex, __onboardingSteps.length);
  document.body.appendChild(tooltip);

  const rect = target.getBoundingClientRect();
  const box = tooltip.querySelector('.onboarding-box');

  const positions = {
    bottom: { top: rect.bottom + 12, left: rect.left + rect.width / 2 - 150 },
    right:  { top: rect.top,         left: rect.right + 12 },
    top:    { top: rect.top - 160,   left: rect.left + rect.width / 2 - 150 },
    left:   { top: rect.top,         left: rect.left - 312 }
  };

  const pos = positions[step.position] || positions.bottom;
  box.style.position = 'fixed';
  box.style.top  = Math.max(10, pos.top) + 'px';
  box.style.left = Math.max(10, Math.min(pos.left, window.innerWidth - 320)) + 'px';
  box.style.zIndex = '10000';
}

function nextOnboardingStep() {
  const step = __onboardingSteps[__onboardingIndex];
  const target = document.querySelector(step?.target);
  if (target) target.classList.remove('onboarding-highlight');

  __onboardingIndex++;
  showOnboardingStep();
}

function skipOnboarding() {
  document.querySelectorAll('.onboarding-highlight')
    .forEach(el => el.classList.remove('onboarding-highlight'));
  const tooltip = document.getElementById('onboarding-tooltip');
  if (tooltip) tooltip.remove();
  finishOnboarding();
}

function finishOnboarding() {
  localStorage.setItem(ONBOARDING_KEY, ONBOARDING_VERSION);
}

window.startOnboarding = startOnboarding;
window.nextOnboardingStep = nextOnboardingStep;
window.skipOnboarding = skipOnboarding;
