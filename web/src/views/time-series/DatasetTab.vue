<script setup lang="ts">
import { computed, ref } from 'vue'
import { storeToRefs } from 'pinia'
import SectionCard from '../../components/SectionCard.vue'
import DialogModal from '../../components/DialogModal.vue'
import FieldControl from '../../components/FieldControl.vue'
import StatusBadge from '../../components/StatusBadge.vue'
import UploadDropzone from '../../components/UploadDropzone.vue'
import WaveformPreviewChart from '../../components/WaveformPreviewChart.vue'
import { useDatasetClassificationStore, type DatasetFile } from '../../stores/datasetClassification'
import { useDatasetRegressionStore } from '../../stores/datasetRegression'

const datasetKinds = [
  {
    id: 'data-classification',
    label: 'Data Classification',
    summary: 'For 1D data where one row is one sample.',
  },
  {
    id: 'regression',
    label: 'Regression',
    summary: 'For 1D data where one row is one sample with numeric targets.',
  },
] as const

const activeKind = ref<(typeof datasetKinds)[number]['id']>('data-classification')
const classificationStore = useDatasetClassificationStore()
const regressionStore = useDatasetRegressionStore()

const classificationState = storeToRefs(classificationStore)
const regressionState = storeToRefs(regressionStore)

const uploadedFolderName = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.uploadedFolderName.value
    : regressionState.uploadedFolderName.value,
)
const numberOfChannels = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.numberOfChannels.value
    : regressionState.numberOfChannels.value,
)
const pendingNumberOfChannels = computed({
  get: () =>
    activeKind.value === 'data-classification'
      ? classificationState.pendingNumberOfChannels.value
      : regressionState.pendingNumberOfChannels.value,
  set: (value: string) => {
    if (activeKind.value === 'data-classification') {
      classificationState.pendingNumberOfChannels.value = value
    } else {
      regressionState.pendingNumberOfChannels.value = value
    }
  },
})
const pendingTargetColumnStart = computed({
  get: () => regressionState.pendingTargetColumnStart.value,
  set: (value: string) => {
    regressionState.pendingTargetColumnStart.value = value
  },
})
const pendingTargetColumnEnd = computed({
  get: () => regressionState.pendingTargetColumnEnd.value,
  set: (value: string) => {
    regressionState.pendingTargetColumnEnd.value = value
  },
})
const isChannelDialogOpen = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.isChannelDialogOpen.value
    : regressionState.isChannelDialogOpen.value,
)
const previewSelection = computed({
  get: () =>
    activeKind.value === 'data-classification'
      ? classificationState.previewCategory.value
      : regressionState.previewSplit.value,
  set: (value: string) => {
    if (activeKind.value === 'data-classification') {
      classificationState.previewCategory.value = value
    } else {
      regressionState.previewSplit.value = value
    }
  },
})
const previewOptions = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.categoryOptions.value
    : regressionState.previewOptions.value,
)
const previewRowValues = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.previewRowValues.value
    : regressionState.previewRowValues.value,
)
const previewTargetValues = computed(() =>
  activeKind.value === 'regression' ? regressionState.previewTargetValues.value : [],
)
const previewRowSource = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.previewRowSource.value
    : regressionState.previewRowSource.value,
)
const previewError = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.previewError.value
    : regressionState.previewError.value,
)
const isPreviewLoading = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.isPreviewLoading.value
    : regressionState.isPreviewLoading.value,
)
const canConfirmChannels = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.canConfirmChannels.value
    : regressionState.canConfirmChannels.value,
)
const previewDataError = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.previewDataError.value
    : regressionState.previewDataError.value,
)
const waveformChannels = computed(() =>
  activeKind.value === 'data-classification'
    ? classificationState.waveformChannels.value
    : regressionState.waveformChannels.value,
)

function handleKindChange(kindId: (typeof datasetKinds)[number]['id']) {
  activeKind.value = kindId
}

function handleFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  const files = Array.from(target.files ?? []) as DatasetFile[]

  if (activeKind.value === 'data-classification') {
    classificationStore.handleFileSelection(files)
  } else {
    regressionStore.handleFileSelection(files)
  }
}

function handlePreviewSelectionChange(value: string) {
  if (activeKind.value === 'data-classification') {
    classificationStore.setPreviewCategory(value)
  } else {
    regressionStore.setPreviewSplit(value)
  }
}

function handleDialogCancel() {
  if (activeKind.value === 'data-classification') {
    classificationStore.cancelChannelDialog()
  } else {
    regressionStore.cancelChannelDialog()
  }
}

function handleDialogConfirm() {
  if (activeKind.value === 'data-classification') {
    classificationStore.confirmChannelDialog()
  } else {
    regressionStore.confirmChannelDialog()
  }
}

function handleRemoveUpload() {
  if (activeKind.value === 'data-classification') {
    classificationStore.resetDatasetSelection()
  } else {
    regressionStore.resetDatasetSelection()
  }
}

const uploadInputId = 'dataset-upload'
</script>

<template>
  <section class="flex flex-col gap-5 p-6">
    <div class="grid gap-5 xl:grid-cols-2">
      <SectionCard class="h-full">
        <div class="mb-5">
          <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Task Type</p>
          <h3 class="m-0 text-[28px] font-bold leading-[1.2] text-[#0d234a]">
            Choose 1D Task Type
          </h3>
          <p class="mt-3 max-w-[700px] text-[15px] leading-[1.7] text-[#475a77]">
            Choose whether this dataset is used for 1D data classification or 1D
            regression.
          </p>
        </div>

        <div class="grid gap-3.5">
          <button
            v-for="kind in datasetKinds"
            :key="kind.id"
            type="button"
            class="flex items-start gap-3 rounded-[14px] border px-4 py-4 text-left transition"
            :class="
              activeKind === kind.id
                ? 'border-transparent bg-[#0d234a] text-white'
                : 'border-[#d6dfef] bg-white text-[#324869] hover:border-[#0d234a]'
            "
            @click="handleKindChange(kind.id)"
          >
            <span
              class="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border"
              :class="
                activeKind === kind.id
                  ? 'border-white bg-white/15'
                  : 'border-[#9fb3d1] bg-white'
              "
            >
              <span
                class="h-2.5 w-2.5 rounded-full"
                :class="activeKind === kind.id ? 'bg-white' : 'bg-transparent'"
              />
            </span>

            <span class="block flex-1">
              <span class="block text-sm font-semibold">{{ kind.label }}</span>
              <span
                class="mt-2 block text-sm leading-[1.6]"
                :class="activeKind === kind.id ? 'text-[#dce8ff]' : 'text-[#58708f]'"
              >
                {{ kind.summary }}
              </span>
            </span>
          </button>
        </div>
      </SectionCard>

      <SectionCard class="h-full">
        <div class="mb-2 flex items-center justify-between gap-3">
          <p class="m-0 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Upload Folder</p>
          <button
            v-if="uploadedFolderName"
            type="button"
            class="rounded-lg border border-[#f1b4bc] bg-[#fff4f6] px-3 py-1.5 text-xs font-semibold text-[#c24157] transition hover:border-[#dc5168] hover:bg-[#ffe8ec] hover:text-[#a61e36]"
            @click="handleRemoveUpload"
          >
            Remove
          </button>
        </div>
        <div class="flex h-full flex-col">
          <h3 class="m-0 min-h-[68px] text-[28px] font-bold leading-[1.2] text-[#0d234a]">Upload Dataset Folder</h3>
          <p class="mt-3 min-h-[52px] text-[15px] leading-[1.7] text-[#475a77]">
            {{ activeKind === 'data-classification'
              ? 'Upload the dataset root folder for 1D data classification.'
              : 'Upload the dataset root folder for 1D regression.' }}
          </p>

          <UploadDropzone :input-id="uploadInputId" label="Choose Dataset Folder" :text="uploadedFolderName || 'Click to select a dataset root folder.'" />
          <input :id="uploadInputId" class="sr-only" type="file" accept=".csv,text/csv" multiple webkitdirectory directory @change="handleFileChange" />

        </div>
      </SectionCard>
    </div>

    <SectionCard>
      <div class="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Waveform Preview</p>
          <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">
            {{ activeKind === 'data-classification' ? 'Dataset Signal View' : 'Regression Signal View' }}
          </h3>
          <p class="mt-3 max-w-[700px] text-[15px] leading-[1.7] text-[#475a77]">
            {{ activeKind === 'data-classification'
              ? 'Preview one row with interactive chart inspection. Hover shows current point values, and wheel or slider zoom inspects local detail.'
              : 'Preview one row of input features and inspect the target value separately.' }}
          </p>
        </div>
        <div class="flex flex-wrap gap-2.5">
          <StatusBadge>Type: {{ activeKind === 'data-classification' ? 'Data Classification' : 'Regression' }}</StatusBadge>
          <StatusBadge>Channels: {{ numberOfChannels || 'Pending' }}</StatusBadge>
        </div>
      </div>

      <div class="mt-5 rounded-[20px] border border-[#d6dfef] bg-white p-4 md:p-5">
        <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <FieldControl :label="activeKind === 'data-classification' ? 'Category' : 'Dataset Split'" class="max-w-xs">
            <select :value="previewSelection" class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]" @change="handlePreviewSelectionChange(($event.target as HTMLSelectElement).value)">
              <option value="" disabled>{{ activeKind === 'data-classification' ? 'Select category' : 'Select sample group' }}</option>
              <option v-for="option in previewOptions" :key="option.key" :value="option.key">{{ option.label }}</option>
            </select>
          </FieldControl>
          <div class="flex flex-wrap gap-2.5">
            <StatusBadge v-if="previewRowSource" tone="muted">Source: {{ previewRowSource }}</StatusBadge>
            <StatusBadge v-if="previewRowValues.length" tone="muted">Row Width: {{ previewRowValues.length }}</StatusBadge>
            <StatusBadge v-if="activeKind === 'regression' && previewTargetValues.length" tone="muted">Target: {{ previewTargetValues.join(', ') }}</StatusBadge>
          </div>
        </div>

        <div v-if="isPreviewLoading || previewError || previewDataError" class="mt-5 flex min-h-80 items-center justify-center rounded-[16px] border border-dashed border-[#c9d7ea] bg-[#fafafa] px-6 text-center text-sm leading-[1.7] text-[#58708f]">
          {{ isPreviewLoading ? (activeKind === 'data-classification' ? 'Loading one row from the selected category...' : 'Loading one row from the selected split...') : previewError || previewDataError }}
        </div>
        <div v-else class="mt-5">
          <WaveformPreviewChart :channels="waveformChannels" />
        </div>
      </div>
    </SectionCard>

    <DialogModal :open="isChannelDialogOpen" @close="handleDialogCancel">
      <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Dataset Config</p>
      <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">Set Number of Channels</h3>
      <p class="mt-3 text-[15px] leading-[1.7] text-[#475a77]">
        {{ activeKind === 'data-classification'
          ? 'Each row is one sample. Enter how many channels are interleaved inside the row, for example `x1 y1 z1 x2 y2 z2 ...` means `3` channels.'
          : 'Each row contains input features and one or more target columns. Enter how many input channels are interleaved in the input part, then set the target column range using 1-based column numbers.' }}
      </p>
      <FieldControl label="Number of Channels" class="mt-6">
        <input v-model="pendingNumberOfChannels" class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]" type="number" min="1" :placeholder="activeKind === 'data-classification' ? 'For example: 3' : 'For example: 21'" />
      </FieldControl>
      <div v-if="activeKind === 'regression'" class="mt-6 grid gap-4 md:grid-cols-2">
        <FieldControl label="Target Column Start">
          <input v-model="pendingTargetColumnStart" class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]" type="number" min="1" placeholder="For example: 22" />
        </FieldControl>
        <FieldControl label="Target Column End">
          <input v-model="pendingTargetColumnEnd" class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]" type="number" min="1" placeholder="For example: 22" />
        </FieldControl>
      </div>
      <div class="mt-6 flex justify-end gap-3">
        <button type="button" class="rounded-xl border border-[#c9d7ea] px-4 py-2.5 text-sm font-semibold text-[#324869] transition hover:border-[#0d234a]" @click="handleDialogCancel">Cancel</button>
        <button type="button" class="rounded-xl bg-[#0d234a] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#132f63] disabled:cursor-not-allowed disabled:bg-[#c9d7ea]" :disabled="!canConfirmChannels" @click="handleDialogConfirm">Confirm</button>
      </div>
    </DialogModal>
  </section>
</template>
