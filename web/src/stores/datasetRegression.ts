import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import type { DatasetFile } from './datasetClassification'

interface PreviewOption {
  key: 'train' | 'test'
  label: string
  previewFile: DatasetFile
}

const waveformPalette = ['#0f172a', '#0f766e', '#f59e0b', '#2563eb', '#e11d48', '#7c3aed']

function getRelativeSegments(file?: DatasetFile) {
  if (!file) return []
  return (file.webkitRelativePath || file.name || '').split(/[\\/]/).filter(Boolean)
}

function getRegressionSection(file: DatasetFile) {
  const segments = getRelativeSegments(file)
  return segments.find((segment) => segment === 'train' || segment === 'test') ?? ''
}

function getDatasetFileName(file: DatasetFile) {
  const segments = getRelativeSegments(file)
  return segments[segments.length - 1] ?? file.name
}

export const useDatasetRegressionStore = defineStore('dataset-regression', () => {
  const uploadedDatasetFiles = ref<DatasetFile[]>([])
  const uploadedFolderName = ref('')
  const uploadedFileCount = ref(0)
  const trainFileCount = ref(0)
  const testFileCount = ref(0)
  const numberOfChannels = ref('')
  const pendingNumberOfChannels = ref('')
  const targetColumnStart = ref('22')
  const targetColumnEnd = ref('22')
  const pendingTargetColumnStart = ref('22')
  const pendingTargetColumnEnd = ref('22')
  const isChannelDialogOpen = ref(false)
  const previewSplit = ref('')
  const previewRowValues = ref<number[]>([])
  const previewTargetValues = ref<number[]>([])
  const previewRowSource = ref('')
  const previewError = ref('')
  const isPreviewLoading = ref(false)

  const canConfirmChannels = computed(() => {
    const channels = Number(pendingNumberOfChannels.value)
    const start = Number(pendingTargetColumnStart.value)
    const end = Number(pendingTargetColumnEnd.value)
    return (
      Number.isFinite(channels) &&
      channels > 0 &&
      Number.isFinite(start) &&
      start > 0 &&
      Number.isFinite(end) &&
      end > 0 &&
      start <= end
    )
  })

  const previewOptions = computed<PreviewOption[]>(() => {
    const options: PreviewOption[] = []
    const trainFile = uploadedDatasetFiles.value.find((file) => getRegressionSection(file) === 'train')
    const testFile = uploadedDatasetFiles.value.find((file) => getRegressionSection(file) === 'test')
    if (trainFile) options.push({ key: 'train', label: 'Train', previewFile: trainFile })
    if (testFile) options.push({ key: 'test', label: 'Test', previewFile: testFile })
    return options
  })

  const configuredChannels = computed(() => {
    const parsed = Number(numberOfChannels.value)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
  })

  const configuredTargetColumnStart = computed(() => {
    const parsed = Number(targetColumnStart.value)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
  })

  const configuredTargetColumnEnd = computed(() => {
    const parsed = Number(targetColumnEnd.value)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
  })

  const targetColumnSummary = computed(() => {
    if (!configuredTargetColumnStart.value || !configuredTargetColumnEnd.value) return 'Not set yet'
    if (configuredTargetColumnStart.value === configuredTargetColumnEnd.value) {
      return `${configuredTargetColumnStart.value}`
    }
    return `${configuredTargetColumnStart.value}-${configuredTargetColumnEnd.value}`
  })

  const previewDataError = computed(() => {
    if (!previewRowValues.value.length || !configuredChannels.value) return ''
    if (previewRowValues.value.length % configuredChannels.value !== 0) {
      return `This row has ${previewRowValues.value.length} values and cannot be evenly split into ${configuredChannels.value} channels.`
    }
    return ''
  })

  const waveformChannels = computed(() => {
    if (!previewRowValues.value.length || !configuredChannels.value || previewDataError.value) return []
    if (previewRowValues.value.length === configuredChannels.value) {
      return [
        {
          id: 0,
          label: 'Input Features',
          color: waveformPalette[0],
          values: previewRowValues.value,
        },
      ]
    }
    return Array.from({ length: configuredChannels.value }, (_, index) => ({
      id: index,
      label: `Channel ${index + 1}`,
      color: waveformPalette[index % waveformPalette.length],
      values: previewRowValues.value.filter((_, valueIndex) => valueIndex % configuredChannels.value === index),
    }))
  })

  function resetPreviewState() {
    previewSplit.value = ''
    previewRowValues.value = []
    previewTargetValues.value = []
    previewRowSource.value = ''
    previewError.value = ''
    isPreviewLoading.value = false
  }

  function resetDatasetSelection() {
    uploadedDatasetFiles.value = []
    uploadedFolderName.value = ''
    uploadedFileCount.value = 0
    trainFileCount.value = 0
    testFileCount.value = 0
    numberOfChannels.value = ''
    pendingNumberOfChannels.value = ''
    targetColumnStart.value = '22'
    targetColumnEnd.value = '22'
    pendingTargetColumnStart.value = '22'
    pendingTargetColumnEnd.value = '22'
    isChannelDialogOpen.value = false
    resetPreviewState()
  }

  function handleFileSelection(files: DatasetFile[]) {
    if (!files.length) {
      resetDatasetSelection()
      return
    }

    uploadedDatasetFiles.value = files
    uploadedFolderName.value = getRelativeSegments(files[0])[0] ?? ''
    uploadedFileCount.value = files.length
    trainFileCount.value = files.filter((file) => getRegressionSection(file) === 'train').length
    testFileCount.value = files.filter((file) => getRegressionSection(file) === 'test').length
    resetPreviewState()
    pendingNumberOfChannels.value = numberOfChannels.value
    pendingTargetColumnStart.value = targetColumnStart.value
    pendingTargetColumnEnd.value = targetColumnEnd.value
    isChannelDialogOpen.value = true
  }

  async function loadPreviewRow() {
    const selectedOption = previewOptions.value.find((option) => option.key === previewSplit.value)
    if (!selectedOption?.previewFile) {
      resetPreviewState()
      return
    }

    isPreviewLoading.value = true
    previewError.value = ''
    try {
      const content = await selectedOption.previewFile.text()
      const line = content.split(/\r?\n/).map((row) => row.trim()).find((row) => row.length > 0)
      if (!line) throw new Error('The selected regression file is empty.')
      const tokens = line.split(/[,\s]+/).filter(Boolean)
      const values = tokens.map((token) => Number(token))
      if (values.some((value) => Number.isNaN(value))) {
        throw new Error('The selected row contains non-numeric values.')
      }
      const targetStartIndex = configuredTargetColumnStart.value - 1
      const targetEndIndex = configuredTargetColumnEnd.value - 1
      if (targetStartIndex < 0 || targetEndIndex < targetStartIndex) {
        throw new Error('The target column range is invalid.')
      }
      if (targetEndIndex >= values.length) {
        throw new Error(`This row has ${values.length} columns, so target columns ${targetColumnSummary.value} are out of range.`)
      }

      const inputValues = values.filter((_, index) => index < targetStartIndex || index > targetEndIndex)
      const targetValues = values.slice(targetStartIndex, targetEndIndex + 1)
      if (!inputValues.length) {
        throw new Error('At least one input column is required outside the target column range.')
      }

      previewRowValues.value = inputValues
      previewTargetValues.value = targetValues
      previewRowSource.value = getDatasetFileName(selectedOption.previewFile)
    } catch (error) {
      previewRowValues.value = []
      previewTargetValues.value = []
      previewRowSource.value = ''
      previewError.value = error instanceof Error ? error.message : 'Failed to read the preview row.'
    } finally {
      isPreviewLoading.value = false
    }
  }

  async function setPreviewSplit(split: string) {
    previewSplit.value = split
    if (!previewSplit.value || !configuredChannels.value) return
    await loadPreviewRow()
  }

  async function confirmChannelDialog() {
    if (!canConfirmChannels.value) return
    numberOfChannels.value = pendingNumberOfChannels.value
    targetColumnStart.value = pendingTargetColumnStart.value
    targetColumnEnd.value = pendingTargetColumnEnd.value
    isChannelDialogOpen.value = false
    if (!previewOptions.value.some((option) => option.key === previewSplit.value)) {
      previewSplit.value = previewOptions.value[0]?.key ?? ''
    }
    await loadPreviewRow()
  }

  function cancelChannelDialog() {
    resetDatasetSelection()
  }

  return {
    uploadedFolderName,
    uploadedFileCount,
    trainFileCount,
    testFileCount,
    numberOfChannels,
    pendingNumberOfChannels,
    targetColumnStart,
    targetColumnEnd,
    pendingTargetColumnStart,
    pendingTargetColumnEnd,
    isChannelDialogOpen,
    previewSplit,
    previewRowValues,
    previewTargetValues,
    previewRowSource,
    previewError,
    isPreviewLoading,
    canConfirmChannels,
    previewOptions,
    previewDataError,
    targetColumnSummary,
    waveformChannels,
    handleFileSelection,
    setPreviewSplit,
    confirmChannelDialog,
    cancelChannelDialog,
    resetDatasetSelection,
  }
})
