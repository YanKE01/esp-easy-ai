import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

interface HiddenLayerConfig {
  id: string
  units: string
  removable: boolean
}

interface CnnBlockLayer {
  id: string
  type: 'cnn-block'
  filters: string
  kernelSize: string
  poolSize: string
  removable: boolean
}

interface CnnDenseLayer {
  id: string
  type: 'dense'
  units: string
  removable: boolean
}

interface CnnDropoutLayer {
  id: string
  type: 'dropout'
  rate: string
  removable: boolean
}

interface CnnGapLayer {
  id: string
  type: 'gap'
  removable: boolean
}

type CnnLayer = CnnBlockLayer | CnnDenseLayer | CnnDropoutLayer | CnnGapLayer

let layerSequence = 3
let cnnBlockSequence = 3
let cnnDenseSequence = 2
let cnnDropoutSequence = 2
let cnnGapSequence = 2

function createLayer(units: string) {
  const layer = {
    id: `dense-${layerSequence}`,
    units,
    removable: true,
  }

  layerSequence += 1
  return layer
}

function createCnnBlock(filters: string, kernelSize: string, poolSize: string): CnnBlockLayer {
  const block = {
    id: `cnn-block-${cnnBlockSequence}`,
    type: 'cnn-block' as const,
    filters,
    kernelSize,
    poolSize,
    removable: true,
  }

  cnnBlockSequence += 1
  return block
}

function createCnnDenseLayer(units: string): CnnDenseLayer {
  const layer = {
    id: `cnn-dense-${cnnDenseSequence}`,
    type: 'dense' as const,
    units,
    removable: true,
  }

  cnnDenseSequence += 1
  return layer
}

function createCnnDropoutLayer(rate: string): CnnDropoutLayer {
  const layer = {
    id: `cnn-dropout-${cnnDropoutSequence}`,
    type: 'dropout' as const,
    rate,
    removable: true,
  }

  cnnDropoutSequence += 1
  return layer
}

function createCnnGapLayer(): CnnGapLayer {
  const layer = {
    id: `cnn-gap-${cnnGapSequence}`,
    type: 'gap' as const,
    removable: true,
  }

  cnnGapSequence += 1
  return layer
}

function createDefaultCnnLayers(): CnnLayer[] {
  return [
    {
      id: 'cnn-block-1',
      type: 'cnn-block',
      filters: '8',
      kernelSize: '5',
      poolSize: '4',
      removable: true,
    },
    {
      id: 'cnn-block-2',
      type: 'cnn-block',
      filters: '16',
      kernelSize: '5',
      poolSize: '4',
      removable: true,
    },
    {
      id: 'cnn-gap-1',
      type: 'gap',
      removable: true,
    },
    {
      id: 'cnn-dense-1',
      type: 'dense',
      units: '32',
      removable: true,
    },
    {
      id: 'cnn-dropout-1',
      type: 'dropout',
      rate: '0.2',
      removable: true,
    },
  ]
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
  const cnnLayers = ref<CnnLayer[]>(createDefaultCnnLayers())
  const splitTrainPercent = ref('80')
  const splitTestPercent = ref('20')
  const splitSeed = ref('42')
  const isSplitDialogOpen = ref(false)
  const editingLayerId = ref('')
  const editingLayerUnits = ref('')
  const isLayerDialogOpen = ref(false)
  const editingCnnBlockId = ref('')
  const editingCnnFilters = ref('')
  const editingCnnKernelSize = ref('')
  const editingCnnPoolSize = ref('')
  const isCnnBlockDialogOpen = ref(false)
  const editingCnnDenseId = ref('')
  const editingCnnDenseUnits = ref('')
  const isCnnDenseDialogOpen = ref(false)
  const editingCnnDropoutId = ref('')
  const editingCnnDropoutRate = ref('')
  const isCnnDropoutDialogOpen = ref(false)

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

  function openCnnBlockDialog(layerId: string) {
    const layer = cnnLayers.value.find((item) => item.id === layerId && item.type === 'cnn-block')
    if (!layer || layer.type !== 'cnn-block') {
      return
    }

    editingCnnBlockId.value = layer.id
    editingCnnFilters.value = layer.filters
    editingCnnKernelSize.value = layer.kernelSize
    editingCnnPoolSize.value = layer.poolSize
    isCnnBlockDialogOpen.value = true
  }

  function closeCnnBlockDialog() {
    editingCnnBlockId.value = ''
    editingCnnFilters.value = ''
    editingCnnKernelSize.value = ''
    editingCnnPoolSize.value = ''
    isCnnBlockDialogOpen.value = false
  }

  function confirmCnnBlockDialog() {
    if (!editingCnnBlockId.value) {
      return
    }

    const layer = cnnLayers.value.find((item) => item.id === editingCnnBlockId.value && item.type === 'cnn-block')
    if (!layer || layer.type !== 'cnn-block') {
      return
    }

    layer.filters = editingCnnFilters.value
    layer.kernelSize = editingCnnKernelSize.value
    layer.poolSize = editingCnnPoolSize.value
    closeCnnBlockDialog()
  }

  function addCnnBlock() {
    cnnLayers.value.push(createCnnBlock('16', '5', '4'))
  }

  function openCnnDenseDialog(layerId: string) {
    const layer = cnnLayers.value.find((item) => item.id === layerId && item.type === 'dense')
    if (!layer || layer.type !== 'dense') {
      return
    }

    editingCnnDenseId.value = layer.id
    editingCnnDenseUnits.value = layer.units
    isCnnDenseDialogOpen.value = true
  }

  function closeCnnDenseDialog() {
    editingCnnDenseId.value = ''
    editingCnnDenseUnits.value = ''
    isCnnDenseDialogOpen.value = false
  }

  function confirmCnnDenseDialog() {
    if (!editingCnnDenseId.value) {
      return
    }

    const layer = cnnLayers.value.find((item) => item.id === editingCnnDenseId.value && item.type === 'dense')
    if (!layer || layer.type !== 'dense') {
      return
    }

    layer.units = editingCnnDenseUnits.value
    closeCnnDenseDialog()
  }

  function addCnnDenseLayer() {
    cnnLayers.value.push(createCnnDenseLayer('32'))
  }

  function openCnnDropoutDialog(layerId: string) {
    const layer = cnnLayers.value.find((item) => item.id === layerId && item.type === 'dropout')
    if (!layer || layer.type !== 'dropout') {
      return
    }

    editingCnnDropoutId.value = layer.id
    editingCnnDropoutRate.value = layer.rate
    isCnnDropoutDialogOpen.value = true
  }

  function closeCnnDropoutDialog() {
    editingCnnDropoutId.value = ''
    editingCnnDropoutRate.value = ''
    isCnnDropoutDialogOpen.value = false
  }

  function confirmCnnDropoutDialog() {
    if (!editingCnnDropoutId.value) {
      return
    }

    const layer = cnnLayers.value.find((item) => item.id === editingCnnDropoutId.value && item.type === 'dropout')
    if (!layer || layer.type !== 'dropout') {
      return
    }

    layer.rate = editingCnnDropoutRate.value
    closeCnnDropoutDialog()
  }

  function addCnnDropoutLayer() {
    cnnLayers.value.push(createCnnDropoutLayer('0.2'))
  }

  function removeCnnLayer(layerId: string) {
    cnnLayers.value = cnnLayers.value.filter((layer) => layer.id !== layerId)
  }

  function restoreCnnPreset() {
    cnnLayers.value = createDefaultCnnLayers()
    cnnBlockSequence = 3
    cnnGapSequence = 2
    cnnDenseSequence = 2
    cnnDropoutSequence = 2
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
    cnnLayers,
    splitTrainPercent,
    splitTestPercent,
    splitSeed,
    isSplitDialogOpen,
    editingLayerId,
    editingLayerUnits,
    isLayerDialogOpen,
    editingCnnBlockId,
    editingCnnFilters,
    editingCnnKernelSize,
    editingCnnPoolSize,
    isCnnBlockDialogOpen,
    editingCnnDenseId,
    editingCnnDenseUnits,
    isCnnDenseDialogOpen,
    editingCnnDropoutId,
    editingCnnDropoutRate,
    isCnnDropoutDialogOpen,
    splitConfigError,
    setLayerUnits,
    openLayerDialog,
    closeLayerDialog,
    confirmLayerDialog,
    addDenseLayer,
    removeDenseLayer,
    openCnnBlockDialog,
    closeCnnBlockDialog,
    confirmCnnBlockDialog,
    addCnnBlock,
    openCnnDenseDialog,
    closeCnnDenseDialog,
    confirmCnnDenseDialog,
    addCnnDenseLayer,
    openCnnDropoutDialog,
    closeCnnDropoutDialog,
    confirmCnnDropoutDialog,
    addCnnDropoutLayer,
    removeCnnLayer,
    restoreCnnPreset,
    openSplitDialog,
    closeSplitDialog,
  }
})
