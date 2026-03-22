<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import WaveformPreviewChart from '../../components/WaveformPreviewChart.vue'

type DatasetFile = File & {
  webkitRelativePath?: string
}

const datasetKinds = [
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

const waveformPalette = ['#0d234a', '#2573d6', '#f29f05', '#11a36a', '#d9485f', '#11a36a']

const activeKind = ref('data-classification')
const fileInputRef = ref<HTMLInputElement | null>(null)
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

function getRelativeSegments(file: DatasetFile) {
  return (file.webkitRelativePath || file.name).split('/').filter(Boolean)
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

function handleFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  const files = Array.from(target.files ?? []) as DatasetFile[]

  if (activeKind.value === 'data-classification') {
    uploadedDatasetFiles.value = files
    uploadedFolderName.value = getRelativeSegments(files[0] as DatasetFile)[0] ?? ''
    uploadedFileCount.value = files.length
    trainFileCount.value = files.filter((file) => getDatasetSection(file) === 'train').length
    testFileCount.value = files.filter((file) => getDatasetSection(file) === 'test').length
    uploadedFileName.value = ''
    previewCategory.value = ''
    previewRowValues.value = []
    previewRowSource.value = ''
    previewError.value = ''
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

const isDataClassification = computed(() => activeKind.value === 'data-classification')
const currentKind = computed(() => datasetKinds.find((kind) => kind.id === activeKind.value))
const canConfirmChannels = computed(() => Number(pendingNumberOfChannels.value) > 0)

const categoryOptions = computed(() => {
  const categoryMap = new Map<
    string,
    { key: string; label: string; trainFile?: DatasetFile; testFile?: DatasetFile }
  >()

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
    values: previewRowValues.value.filter((_, valueIndex) => valueIndex % configuredChannels.value === index),
  }))
})

async function loadPreviewRow() {
  const selectedCategory = categoryOptions.value.find((option) => option.key === previewCategory.value)

  if (!selectedCategory?.previewFile) {
    previewRowValues.value = []
    previewRowSource.value = ''
    previewError.value = ''
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

function confirmChannelDialog() {
  if (!canConfirmChannels.value) {
    return
  }

  numberOfChannels.value = pendingNumberOfChannels.value
  isChannelDialogOpen.value = false

  if (!categoryOptions.value.some((option) => option.key === previewCategory.value)) {
    previewCategory.value = categoryOptions.value[0]?.key ?? ''
  }

  void loadPreviewRow()
}

function cancelChannelDialog() {
  isChannelDialogOpen.value = false
  pendingNumberOfChannels.value = ''
  uploadedDatasetFiles.value = []
  uploadedFolderName.value = ''
  uploadedFileCount.value = 0
  trainFileCount.value = 0
  testFileCount.value = 0
  numberOfChannels.value = ''
  previewCategory.value = ''
  previewRowValues.value = []
  previewRowSource.value = ''
  previewError.value = ''

  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

watch(previewCategory, () => {
  if (!previewCategory.value || !configuredChannels.value) {
    return
  }

  void loadPreviewRow()
})
</script>

<template>
  <section class="dataset-panel">
    <div class="dataset-grid">
      <article class="dataset-card">
        <div class="section-heading">
          <p class="section-eyebrow">Dataset Setup</p>
          <h2 class="section-title">Choose 1D Data Type</h2>
          <p class="section-description">
            Set the dataset mode here first. The later training flow should inherit this
            choice instead of asking again.
          </p>
        </div>

        <div class="mode-options">
          <button
            v-for="kind in datasetKinds"
            :key="kind.id"
            class="mode-option"
            :class="{ 'is-active': activeKind === kind.id }"
            type="button"
            @click="activeKind = kind.id"
          >
            <span class="mode-label">{{ kind.label }}</span>
            <span class="mode-summary">{{ kind.summary }}</span>
          </button>
        </div>

        <p v-if="isDataClassification" class="field-help dataset-setup-note">
          Upload the dataset folder first. After selection, a dialog will ask for
          `Number of Channels`, then the later flow can continue.
        </p>
      </article>

      <article class="dataset-card">
        <p class="section-eyebrow">
          {{ isDataClassification ? 'Upload Folder' : 'Upload CSV' }}
        </p>
        <h3 class="selection-title">
          {{ isDataClassification ? 'Upload Dataset Folder' : 'Dataset File Entry' }}
        </h3>
        <p class="selection-summary">
          <template v-if="isDataClassification">
            Upload the dataset root folder. The folder must contain exact subdirectories
            named `train` and `test`, matching the sample dataset layout.
          </template>
          <template v-else>
            Upload the source CSV here after picking the 1D data type. The later column
            mapping and preview should work on the same file.
          </template>
        </p>

        <label class="upload-dropzone" for="dataset-upload">
          <span class="upload-title">
            {{ isDataClassification ? 'Choose Dataset Folder' : 'Choose CSV File' }}
          </span>
          <span class="upload-text">
            <template v-if="isDataClassification">
              {{ uploadedFolderName || 'Click to select the dataset root folder that contains train/ and test/' }}
            </template>
            <template v-else>
              {{ uploadedFileName || 'Click to select a local .csv dataset file' }}
            </template>
          </span>
        </label>

        <input
          id="dataset-upload"
          ref="fileInputRef"
          class="upload-input"
          type="file"
          :accept="isDataClassification ? '.csv,text/csv' : '.csv,text/csv'"
          :multiple="isDataClassification"
          :webkitdirectory="isDataClassification"
          :directory="isDataClassification"
          @change="handleFileChange"
        />

        <template v-if="isDataClassification">
          <ul class="selection-list">
            <li>Required directories: `train` and `test`.</li>
            <li>Current type: {{ currentKind?.label }}</li>
            <li>Number of channels: {{ numberOfChannels || 'Not set yet' }}</li>
            <li v-if="uploadedFolderName">
              Selected folder: {{ uploadedFolderName }} | Files: {{ uploadedFileCount }}
            </li>
            <li v-if="uploadedFolderName">
              Train files: {{ trainFileCount }} | Test files: {{ testFileCount }}
            </li>
          </ul>
        </template>

        <ul v-else class="selection-list">
          <li>Accepted format: CSV wide-table input.</li>
          <li>Keep the upload step inside Dataset so Training does not ask again.</li>
          <li>Current type: {{ currentKind?.label }}</li>
        </ul>
      </article>
    </div>

    <article class="dataset-card waveform-card">
      <div class="waveform-header">
        <div>
          <p class="section-eyebrow">Waveform Preview</p>
          <h3 class="selection-title">Dataset Signal View</h3>
          <p class="selection-summary waveform-summary">
            Preview one row with framework-based chart interaction. Hover shows current
            point values, and wheel or slider zoom can inspect local detail.
          </p>
        </div>

        <div class="waveform-meta">
          <span class="waveform-badge">Type: {{ currentKind?.label }}</span>
          <span class="waveform-badge">Channels: {{ numberOfChannels || 'Pending' }}</span>
        </div>
      </div>

      <div class="waveform-shell">
        <div class="waveform-toolbar">
          <label class="field-block waveform-field">
            <span class="field-label">Category</span>
            <select v-model="previewCategory" class="field-input waveform-select">
              <option value="" disabled>Select category</option>
              <option
                v-for="option in categoryOptions"
                :key="option.key"
                :value="option.key"
              >
                {{ option.label }}
              </option>
            </select>
          </label>

          <div class="waveform-meta waveform-meta-compact">
            <span v-if="previewRowSource" class="waveform-badge">Source: {{ previewRowSource }}</span>
            <span v-if="previewRowValues.length" class="waveform-badge">Row Width: {{ previewRowValues.length }}</span>
          </div>
        </div>

        <div v-if="isPreviewLoading" class="waveform-empty">
          <p class="waveform-footnote">Loading one row from the selected category...</p>
        </div>
        <div v-else-if="previewError" class="waveform-empty">
          <p class="waveform-footnote">{{ previewError }}</p>
        </div>
        <div v-else-if="previewDataError" class="waveform-empty">
          <p class="waveform-footnote">{{ previewDataError }}</p>
        </div>
        <WaveformPreviewChart v-else :channels="waveformChannels" />
      </div>
    </article>

    <div v-if="isChannelDialogOpen" class="dialog-backdrop">
      <div class="dialog-card">
        <p class="section-eyebrow">Dataset Config</p>
        <h3 class="selection-title">Set Number of Channels</h3>
        <p class="selection-summary">
          Each row is one sample. Enter how many channels are interleaved inside the row,
          for example `x1 y1 z1 x2 y2 z2 ...` means `3` channels.
        </p>

        <label class="field-block">
          <span class="field-label">Number of Channels</span>
          <input
            v-model="pendingNumberOfChannels"
            class="field-input"
            type="number"
            min="1"
            placeholder="For example: 3"
          />
        </label>

        <div class="dialog-actions">
          <button class="dialog-button dialog-button-secondary" type="button" @click="cancelChannelDialog">
            Cancel
          </button>
          <button
            class="dialog-button dialog-button-primary"
            type="button"
            :disabled="!canConfirmChannels"
            @click="confirmChannelDialog"
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
