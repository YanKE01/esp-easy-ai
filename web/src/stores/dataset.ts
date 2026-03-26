import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export type DatasetFile = File & {
  webkitRelativePath?: string
}

interface DatasetKind {
  id: 'time-series' | 'data-classification'
  label: string
  summary: string
}

interface CategoryOption {
  key: string
  label: string
  trainFile?: DatasetFile
  testFile?: DatasetFile
  previewFile?: DatasetFile
}

const datasetKinds: DatasetKind[] = [
  {
    id: 'time-series',
    label: 'Time Series',
    summary: 'For continuous sequences where one series is one sample.',
  },
  {
    id: 'data-classification',
    label: 'Data Classification',
    summary: 'For non-sequence 1D data where one row is one sample.',
  },
]

const waveformPalette = ['#0f172a', '#0f766e', '#f59e0b', '#2563eb', '#e11d48', '#7c3aed']

function getRelativeSegments(file?: DatasetFile) {
  if (!file) {
    return []
  }

  return (file.webkitRelativePath || file.name || '').split('/').filter(Boolean)
}

function getDatasetSection(file: DatasetFile) {
  return getRelativeSegments(file)[1] ?? ''
}

function getDatasetFileName(file: DatasetFile) {
  const segments = getRelativeSegments(file)
  return segments[segments.length - 1] ?? file.name
}

function getCategoryKey(file: DatasetFile) {
  const section = getDatasetSection(file)
  const fileName = getDatasetFileName(file)
  const baseName = fileName.replace(/\.[^.]+$/, '')

  if (section === 'train' && baseName.startsWith('train_')) {
    return baseName.slice(6)
  }

  if (section === 'test' && baseName.startsWith('test_')) {
    return baseName.slice(5)
  }

  return baseName
}

export const useDatasetStore = defineStore('dataset', () => {
  const activeKind = ref<DatasetKind['id']>('data-classification')
  const uploadedDatasetFiles = ref<DatasetFile[]>([])
  const uploadedFileName = ref('')
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

  const isDataClassification = computed(() => activeKind.value === 'data-classification')
  const currentKind = computed(() => datasetKinds.find((kind) => kind.id === activeKind.value))
  const canConfirmChannels = computed(() => Number(pendingNumberOfChannels.value) > 0)

  const categoryOptions = computed<CategoryOption[]>(() => {
    const categoryMap = new Map<string, CategoryOption>()

    uploadedDatasetFiles.value.forEach((file) => {
      const section = getDatasetSection(file)

      if (section !== 'train' && section !== 'test') {
        return
      }

      const key = getCategoryKey(file)
      const existing = categoryMap.get(key) ?? { key, label: key }

      if (section === 'train') {
        existing.trainFile = file
      } else {
        existing.testFile = file
      }

      categoryMap.set(key, existing)
    })

    return Array.from(categoryMap.values())
      .sort((left, right) => left.label.localeCompare(right.label))
      .map((option) => ({
        ...option,
        previewFile: option.trainFile ?? option.testFile,
      }))
  })

  const configuredChannels = computed(() => {
    const parsed = Number(numberOfChannels.value)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
  })

  const previewDataError = computed(() => {
    if (!previewRowValues.value.length || !configuredChannels.value) {
      return ''
    }

    if (previewRowValues.value.length % configuredChannels.value !== 0) {
      return `This row has ${previewRowValues.value.length} values and cannot be evenly split into ${configuredChannels.value} channels.`
    }

    return ''
  })

  const waveformChannels = computed(() => {
    if (!previewRowValues.value.length || !configuredChannels.value || previewDataError.value) {
      return []
    }

    return Array.from({ length: configuredChannels.value }, (_, index) => ({
      id: index,
      label: `Channel ${index + 1}`,
      color: waveformPalette[index % waveformPalette.length],
      values: previewRowValues.value.filter(
        (_, valueIndex) => valueIndex % configuredChannels.value === index,
      ),
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
    uploadedFileName.value = ''
    uploadedFolderName.value = ''
    uploadedFileCount.value = 0
    trainFileCount.value = 0
    testFileCount.value = 0
    numberOfChannels.value = ''
    pendingNumberOfChannels.value = ''
    isChannelDialogOpen.value = false
    resetPreviewState()
  }

  function setActiveKind(kindId: DatasetKind['id']) {
    if (activeKind.value === kindId) {
      return
    }

    activeKind.value = kindId
    resetDatasetSelection()
  }

  function handleFileSelection(files: DatasetFile[]) {
    if (!files.length) {
      resetDatasetSelection()
      return
    }

    if (isDataClassification.value) {
      uploadedDatasetFiles.value = files
      uploadedFolderName.value = getRelativeSegments(files[0])[0] ?? ''
      uploadedFileCount.value = files.length
      trainFileCount.value = files.filter((file) => getDatasetSection(file) === 'train').length
      testFileCount.value = files.filter((file) => getDatasetSection(file) === 'test').length
      uploadedFileName.value = ''
      resetPreviewState()
      pendingNumberOfChannels.value = numberOfChannels.value
      isChannelDialogOpen.value = files.length > 0
      return
    }

    uploadedDatasetFiles.value = []
    uploadedFileName.value = files[0]?.name ?? ''
    uploadedFolderName.value = ''
    uploadedFileCount.value = files.length
    trainFileCount.value = 0
    testFileCount.value = 0
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
      const line = content
        .split(/\r?\n/)
        .map((row) => row.trim())
        .find((row) => row.length > 0)

      if (!line) {
        throw new Error('The selected category file is empty.')
      }

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

    if (!previewCategory.value || !configuredChannels.value) {
      return
    }

    await loadPreviewRow()
  }

  async function confirmChannelDialog() {
    if (!canConfirmChannels.value) {
      return
    }

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
    datasetKinds,
    activeKind,
    uploadedFileName,
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
    isDataClassification,
    currentKind,
    canConfirmChannels,
    categoryOptions,
    configuredChannels,
    previewDataError,
    waveformChannels,
    setActiveKind,
    handleFileSelection,
    setPreviewCategory,
    confirmChannelDialog,
    cancelChannelDialog,
  }
})
