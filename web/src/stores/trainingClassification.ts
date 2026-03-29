import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

interface HiddenLayerConfig {
  id: string
  units: string
  removable: boolean
}

let layerSequence = 3

function createLayer(units: string) {
  const layer = {
    id: `dense-${layerSequence}`,
    units,
    removable: true,
  }

  layerSequence += 1
  return layer
}

export const useTrainingClassificationStore = defineStore('training-classification', () => {
  const modelPreset = ref<'mlp' | 'cnn'>('mlp')
  const epochs = ref('30')
  const batchSize = ref('32')
  const learningRate = ref('0.001')
  const optimizer = ref<'adam' | 'adamw' | 'sgd'>('adam')
  const inputNormalization = ref<'none' | 'z_score' | 'min_max'>('none')
  const hiddenLayers = ref<HiddenLayerConfig[]>([
    { id: 'dense-1', units: '128', removable: false },
    { id: 'dense-2', units: '64', removable: false },
  ])
  const splitTrainPercent = ref('80')
  const splitTestPercent = ref('20')
  const splitSeed = ref('42')
  const isSplitDialogOpen = ref(false)
  const editingLayerId = ref('')
  const editingLayerUnits = ref('')
  const isLayerDialogOpen = ref(false)

  const splitConfigError = computed(() => {
    const train = Number(splitTrainPercent.value)
    const test = Number(splitTestPercent.value)
    const seed = Number(splitSeed.value)

    if (!Number.isFinite(train) || !Number.isFinite(test) || train <= 0 || test <= 0) {
      return 'Train and test percentages must both be positive numbers.'
    }

    if (train + test !== 100) {
      return 'Train and test percentages must add up to 100.'
    }

    if (!Number.isFinite(seed) || seed < 0) {
      return 'Split seed must be a non-negative integer.'
    }

    return ''
  })

  function setLayerUnits(layerId: string, units: string) {
    const layer = hiddenLayers.value.find((item) => item.id === layerId)
    if (layer) {
      layer.units = units
    }
  }

  function openLayerDialog(layerId: string) {
    const layer = hiddenLayers.value.find((item) => item.id === layerId)
    if (!layer) {
      return
    }

    editingLayerId.value = layer.id
    editingLayerUnits.value = layer.units
    isLayerDialogOpen.value = true
  }

  function closeLayerDialog() {
    editingLayerId.value = ''
    editingLayerUnits.value = ''
    isLayerDialogOpen.value = false
  }

  function confirmLayerDialog() {
    if (!editingLayerId.value) {
      return
    }

    setLayerUnits(editingLayerId.value, editingLayerUnits.value)
    closeLayerDialog()
  }

  function addDenseLayer() {
    hiddenLayers.value.push(createLayer('128'))
  }

  function removeDenseLayer(layerId: string) {
    if (hiddenLayers.value.length <= 1) {
      return
    }

    hiddenLayers.value = hiddenLayers.value.filter((layer) => layer.id !== layerId)
  }

  function openSplitDialog() {
    isSplitDialogOpen.value = true
  }

  function closeSplitDialog() {
    isSplitDialogOpen.value = false
  }

  return {
    modelPreset,
    epochs,
    batchSize,
    learningRate,
    optimizer,
    inputNormalization,
    hiddenLayers,
    splitTrainPercent,
    splitTestPercent,
    splitSeed,
    isSplitDialogOpen,
    editingLayerId,
    editingLayerUnits,
    isLayerDialogOpen,
    splitConfigError,
    setLayerUnits,
    openLayerDialog,
    closeLayerDialog,
    confirmLayerDialog,
    addDenseLayer,
    removeDenseLayer,
    openSplitDialog,
    closeSplitDialog,
  }
})
