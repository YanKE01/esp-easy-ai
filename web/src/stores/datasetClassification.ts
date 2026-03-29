import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export type DatasetFile = File & { webkitRelativePath?: string }

interface CategoryOption {
  key: string
  label: string
  trainFile?: DatasetFile
  testFile?: DatasetFile
  previewFile?: DatasetFile
}

type ClassificationDatasetMode = 'pre-split' | 'raw-class-files' | 'invalid'

const waveformPalette = ['#0f172a', '#0f766e', '#f59e0b', '#2563eb', '#e11d48', '#7c3aed']

function getRelativeSegments(file?: DatasetFile) {
  if (!file) return []
  return (file.webkitRelativePath || file.name || '').split(/[\\/]/).filter(Boolean)
}

function getDatasetSection(file: DatasetFile) {
  return getRelativeSegments(file)[1] ?? ''
}

function getDatasetFileName(file: DatasetFile) {
  const segments = getRelativeSegments(file)
  return segments[segments.length - 1] ?? file.name
}

function getCategoryBaseName(file: DatasetFile) {
  return getDatasetFileName(file).replace(/\.[^.]+$/, '')
}

function getCategoryKey(file: DatasetFile, datasetMode: ClassificationDatasetMode) {
  const baseName = getCategoryBaseName(file)

  if (datasetMode === 'pre-split') {
    const section = getDatasetSection(file)
    if (section === 'train' && baseName.startsWith('train_')) return baseName.slice(6)
    if (section === 'test' && baseName.startsWith('test_')) return baseName.slice(5)
  }

  return baseName
}

function detectDatasetMode(files: DatasetFile[]) {
  if (!files.length) {
    return {
      mode: 'invalid' as const,
      error: 'Please upload at least one CSV file.',
    }
  }

  const segmentsList = files.map((file) => getRelativeSegments(file))
  const hasUnsupportedNesting = segmentsList.some((segments) => segments.length < 2 || segments.length > 3)

  if (hasUnsupportedNesting) {
    return {
      mode: 'invalid' as const,
      error:
        'Unsupported folder depth. Use either dataset_root/train/*.csv and dataset_root/test/*.csv, or put class CSV files directly under the dataset root.',
    }
  }

  const hasPreSplitFile = files.some((file) => {
    const section = getDatasetSection(file)
    return section === 'train' || section === 'test'
  })
  const hasRootClassFile = files.some((file) => getRelativeSegments(file).length === 2)

  if (hasPreSplitFile && hasRootClassFile) {
    return {
      mode: 'invalid' as const,
      error:
        'Mixed dataset layouts are not supported. Use either train/test folders or class CSV files directly under the dataset root.',
    }
  }

  if (hasPreSplitFile) {
    const hasInvalidPreSplitFile = files.some((file) => {
      const segments = getRelativeSegments(file)
      return segments.length !== 3 || (segments[1] !== 'train' && segments[1] !== 'test')
    })

    if (hasInvalidPreSplitFile) {
      return {
        mode: 'invalid' as const,
        error:
          'Pre-split classification datasets must contain only train/*.csv and test/*.csv files under the dataset root.',
      }
    }

    const hasTrainFile = files.some((file) => getDatasetSection(file) === 'train')
    const hasTestFile = files.some((file) => getDatasetSection(file) === 'test')

    if (!hasTrainFile || !hasTestFile) {
      return {
        mode: 'invalid' as const,
        error: 'Pre-split classification datasets must include both train and test folders.',
      }
    }

    return {
      mode: 'pre-split' as const,
      error: '',
    }
  }

  const hasInvalidRootClassFile = files.some((file) => getRelativeSegments(file).length !== 2)

  if (hasInvalidRootClassFile) {
    return {
      mode: 'invalid' as const,
      error:
        'Raw classification datasets must place class CSV files directly under the dataset root without extra subfolders.',
    }
  }

  return {
    mode: 'raw-class-files' as const,
    error: '',
  }
}

export const useDatasetClassificationStore = defineStore('dataset-classification', () => {
  const uploadedDatasetFiles = ref<DatasetFile[]>([])
  const uploadedFolderName = ref('')
  const uploadedFileCount = ref(0)
  const trainFileCount = ref(0)
  const testFileCount = ref(0)
  const numberOfChannels = ref('')
  const pendingNumberOfChannels = ref('')
  const isChannelDialogOpen = ref(false)
  const previewCategory = ref('')
  const previewRowValues = ref<number[]>([])
  const previewRowSource = ref('')
  const previewError = ref('')
  const isPreviewLoading = ref(false)
  const datasetMode = ref<ClassificationDatasetMode>('invalid')
  const datasetModeError = ref('')

  const canConfirmChannels = computed(() => Number(pendingNumberOfChannels.value) > 0)

  const categoryOptions = computed<CategoryOption[]>(() => {
    const categoryMap = new Map<string, CategoryOption>()

    uploadedDatasetFiles.value.forEach((file) => {
      if (datasetMode.value === 'pre-split') {
        const section = getDatasetSection(file)
        if (section !== 'train' && section !== 'test') return

        const key = getCategoryKey(file, datasetMode.value)
        const existing = categoryMap.get(key) ?? { key, label: key }

        if (section === 'train') existing.trainFile = file
        else existing.testFile = file

        categoryMap.set(key, existing)
        return
      }

      if (datasetMode.value === 'raw-class-files') {
        const key = getCategoryKey(file, datasetMode.value)
        categoryMap.set(key, {
          key,
          label: key,
          previewFile: file,
          trainFile: file,
        })
      }
    })

    return Array.from(categoryMap.values())
      .sort((left, right) => left.label.localeCompare(right.label))
      .map((option) => ({
        ...option,
        previewFile: option.previewFile ?? option.trainFile ?? option.testFile,
      }))
  })

  const configuredChannels = computed(() => {
    const parsed = Number(numberOfChannels.value)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
  })

  const datasetSummary = computed(() => {
    if (!uploadedDatasetFiles.value.length || datasetMode.value === 'invalid') {
      return 'No valid dataset loaded yet.'
    }

    if (datasetMode.value === 'pre-split') {
      return `Detected ${categoryOptions.value.length} categories from ${trainFileCount.value} train files and ${testFileCount.value} test files.`
    }

    return `Detected ${categoryOptions.value.length} categories from ${uploadedFileCount.value} root-level CSV files. Train/test split will be configured later in the Training stage.`
  })

  const isDatasetReady = computed(() => {
    if (datasetMode.value === 'invalid' || datasetModeError.value) {
      return false
    }

    return categoryOptions.value.length > 0 && configuredChannels.value > 0
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
    return Array.from({ length: configuredChannels.value }, (_, index) => ({
      id: index,
      label: `Channel ${index + 1}`,
      color: waveformPalette[index % waveformPalette.length],
      values: previewRowValues.value.filter((_, valueIndex) => valueIndex % configuredChannels.value === index),
    }))
  })

  function resetPreviewState() {
    previewCategory.value = ''
    previewRowValues.value = []
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
    isChannelDialogOpen.value = false
    datasetMode.value = 'invalid'
    datasetModeError.value = ''
    resetPreviewState()
  }

  function handleFileSelection(files: DatasetFile[]) {
    if (!files.length) {
      resetDatasetSelection()
      return
    }

    const detectedMode = detectDatasetMode(files)

    uploadedDatasetFiles.value = files
    uploadedFolderName.value = getRelativeSegments(files[0])[0] ?? ''
    uploadedFileCount.value = files.length
    datasetMode.value = detectedMode.mode
    datasetModeError.value = detectedMode.error
    trainFileCount.value = files.filter((file) => getDatasetSection(file) === 'train').length
    testFileCount.value = files.filter((file) => getDatasetSection(file) === 'test').length
    resetPreviewState()
    pendingNumberOfChannels.value = numberOfChannels.value
    isChannelDialogOpen.value = !detectedMode.error
  }

  async function loadPreviewRow() {
    const selectedCategory = categoryOptions.value.find((option) => option.key === previewCategory.value)
    if (!selectedCategory?.previewFile) {
      resetPreviewState()
      return
    }

    isPreviewLoading.value = true
    previewError.value = ''
    try {
      const content = await selectedCategory.previewFile.text()
      const line = content.split(/\r?\n/).map((row) => row.trim()).find((row) => row.length > 0)
      if (!line) throw new Error('The selected category file is empty.')
      const tokens = line.split(/[,\s]+/).filter(Boolean)
      const values = tokens.map((token) => Number(token))
      if (values.some((value) => Number.isNaN(value))) {
        throw new Error('The selected row contains non-numeric values.')
      }
      previewRowValues.value = values
      previewRowSource.value = getDatasetFileName(selectedCategory.previewFile)
    } catch (error) {
      previewRowValues.value = []
      previewRowSource.value = ''
      previewError.value = error instanceof Error ? error.message : 'Failed to read the preview row.'
    } finally {
      isPreviewLoading.value = false
    }
  }

  async function setPreviewCategory(category: string) {
    previewCategory.value = category
    if (!previewCategory.value || !configuredChannels.value) return
    await loadPreviewRow()
  }

  async function confirmChannelDialog() {
    if (!canConfirmChannels.value) return
    numberOfChannels.value = pendingNumberOfChannels.value
    isChannelDialogOpen.value = false
    if (!categoryOptions.value.some((option) => option.key === previewCategory.value)) {
      previewCategory.value = categoryOptions.value[0]?.key ?? ''
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
    isChannelDialogOpen,
    previewCategory,
    previewRowValues,
    previewRowSource,
    previewError,
    isPreviewLoading,
    datasetMode,
    datasetModeError,
    canConfirmChannels,
    categoryOptions,
    configuredChannels,
    datasetSummary,
    isDatasetReady,
    previewDataError,
    waveformChannels,
    handleFileSelection,
    setPreviewCategory,
    confirmChannelDialog,
    cancelChannelDialog,
    resetDatasetSelection,
  }
})
