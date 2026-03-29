import { ref } from 'vue'
import { defineStore } from 'pinia'

export type OneDTaskKind = 'data-classification' | 'regression'

export const useOneDWorkflowStore = defineStore('one-d-workflow', () => {
  const activeKind = ref<OneDTaskKind>('data-classification')
  const hasCompletedTraining = ref(false)

  function setActiveKind(kind: OneDTaskKind) {
    if (activeKind.value !== kind) {
      hasCompletedTraining.value = false
    }
    activeKind.value = kind
  }

  function markTrainingCompleted() {
    hasCompletedTraining.value = true
  }

  function resetTrainingProgress() {
    hasCompletedTraining.value = false
  }

  return {
    activeKind,
    hasCompletedTraining,
    setActiveKind,
    markTrainingCompleted,
    resetTrainingProgress,
  }
})
