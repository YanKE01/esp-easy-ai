<script setup lang="ts">
import { computed, watchEffect } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import TabNav from '../components/TabNav.vue'
import { useDatasetClassificationStore } from '../stores/datasetClassification'
import { useDatasetRegressionStore } from '../stores/datasetRegression'
import { useOneDWorkflowStore } from '../stores/oneDWorkflow'

const route = useRoute()
const router = useRouter()
const workflowStore = useOneDWorkflowStore()
const classificationStore = useDatasetClassificationStore()
const regressionStore = useDatasetRegressionStore()

const isDatasetReady = computed(() =>
  workflowStore.activeKind === 'data-classification'
    ? classificationStore.isDatasetReady
    : regressionStore.isDatasetReady,
)

const canOpenTraining = computed(() => isDatasetReady.value)
const canOpenEmulation = computed(() => workflowStore.hasCompletedTraining)

const tabs = computed(() => [
  { label: 'Dataset', to: '/one-d/dataset' },
  { label: 'Training', to: '/one-d/training', disabled: !canOpenTraining.value },
  { label: 'Emulation', to: '/one-d/emulation', disabled: !canOpenEmulation.value },
])

watchEffect(() => {
  if (route.path === '/one-d/training' && !canOpenTraining.value) {
    router.replace('/one-d/dataset')
  }

  if (route.path === '/one-d/emulation' && !canOpenEmulation.value) {
    router.replace(canOpenTraining.value ? '/one-d/training' : '/one-d/dataset')
  }
})
</script>

<template>
  <section class="flex min-h-full flex-col">
    <TabNav :items="tabs" />
    <div class="flex-1">
      <RouterView />
    </div>
  </section>
</template>
