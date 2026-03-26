<script setup lang="ts">
import { ref } from 'vue'
import { storeToRefs } from 'pinia'
import DialogModal from '../../components/DialogModal.vue'
import FieldControl from '../../components/FieldControl.vue'
import SectionCard from '../../components/SectionCard.vue'
import StatusBadge from '../../components/StatusBadge.vue'
import UploadDropzone from '../../components/UploadDropzone.vue'
import WaveformPreviewChart from '../../components/WaveformPreviewChart.vue'
import { useDatasetStore, type DatasetFile } from '../../stores/dataset'

const fileInputRef = ref<HTMLInputElement | null>(null)
const datasetStore = useDatasetStore()

const {
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
  previewDataError,
  waveformChannels,
} = storeToRefs(datasetStore)

function handleFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  const files = Array.from(target.files ?? []) as DatasetFile[]
  datasetStore.handleFileSelection(files)
}

function handleKindChange(kindId: 'time-series' | 'data-classification') {
  datasetStore.setActiveKind(kindId)

  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

function handleDialogCancel() {
  datasetStore.cancelChannelDialog()

  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

const uploadInputId = 'dataset-upload'
</script>

<template>
  <section class="flex flex-col gap-5 p-6">
    <div class="grid gap-5 xl:grid-cols-2">
      <SectionCard>
        <div class="mb-5">
          <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Dataset Setup</p>
          <h3 class="m-0 text-[28px] font-bold leading-[1.2] text-[#0d234a]">
          Choose 1D Data Type
          </h3>
          <p class="mt-3 max-w-[700px] text-[15px] leading-[1.7] text-[#475a77]">
            Set the dataset mode here first. Training and emulation should inherit this
            decision instead of asking again.
          </p>
        </div>

        <div class="grid gap-3.5">
          <button
            v-for="kind in datasetKinds"
            :key="kind.id"
            type="button"
            class="flex flex-col rounded-[14px] border px-4 py-4 text-left transition"
            :class="
              activeKind === kind.id
                ? 'border-transparent bg-[#0d234a] text-white'
                : 'border-[#d6dfef] bg-white text-[#324869] hover:border-[#0d234a]'
            "
            @click="handleKindChange(kind.id)"
          >
            <span class="block text-sm font-semibold">{{ kind.label }}</span>
            <span
              class="mt-2 block text-sm leading-[1.6]"
              :class="activeKind === kind.id ? 'text-[#dce8ff]' : 'text-[#58708f]'"
            >
              {{ kind.summary }}
            </span>
          </button>
        </div>

        <p
          v-if="isDataClassification"
          class="mt-3 text-[14px] leading-[1.7] text-[#58708f]"
        >
          Upload the dataset folder first. After selection, a dialog will ask for
          `Number of Channels`, then later steps can reuse that global configuration.
        </p>
      </SectionCard>

      <SectionCard>
        <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">
          {{ isDataClassification ? 'Upload Folder' : 'Upload CSV' }}
        </p>
        <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">
          {{ isDataClassification ? 'Upload Dataset Folder' : 'Dataset File Entry' }}
        </h3>
        <p class="mt-3 text-[15px] leading-[1.7] text-[#475a77]">
          <template v-if="isDataClassification">
            Upload the dataset root folder. The folder must contain exact subdirectories
            named `train` and `test`, matching the sample dataset layout.
          </template>
          <template v-else>
            Upload the source CSV after picking the 1D data type. Later column mapping
            and preview should work on the same file.
          </template>
        </p>

        <UploadDropzone
          :input-id="uploadInputId"
          :label="isDataClassification ? 'Choose Dataset Folder' : 'Choose CSV File'"
          :text="
            isDataClassification
              ? uploadedFolderName ||
                'Click to select the dataset root folder that contains train/ and test/.'
              : uploadedFileName || 'Click to select a local .csv dataset file.'
          "
        />

        <input
          :id="uploadInputId"
          ref="fileInputRef"
          class="sr-only"
          type="file"
          accept=".csv,text/csv"
          :multiple="isDataClassification"
          :webkitdirectory="isDataClassification"
          :directory="isDataClassification"
          @change="handleFileChange"
        />

        <ul class="mt-5 space-y-2 text-[14px] leading-[1.7] text-[#58708f]">
          <template v-if="isDataClassification">
            <li>
              Required directories: `train` and `test`.
            </li>
            <li>
              Current type: {{ currentKind?.label }}
            </li>
            <li>
              Number of channels: {{ numberOfChannels || 'Not set yet' }}
            </li>
            <li v-if="uploadedFolderName">
              Selected folder: {{ uploadedFolderName }} | Files: {{ uploadedFileCount }}
            </li>
            <li v-if="uploadedFolderName">
              Train files: {{ trainFileCount }} | Test files: {{ testFileCount }}
            </li>
          </template>

          <template v-else>
            <li>
              Accepted format: CSV wide-table input.
            </li>
            <li>
              Keep upload inside Dataset so Training does not ask again.
            </li>
            <li>
              Current type: {{ currentKind?.label }}
            </li>
          </template>
        </ul>
      </SectionCard>
    </div>

    <SectionCard>
      <div class="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">
            Waveform Preview
          </p>
          <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">
            Dataset Signal View
          </h3>
          <p class="mt-3 max-w-[700px] text-[15px] leading-[1.7] text-[#475a77]">
            Preview one row with interactive chart inspection. Hover shows current
            point values, and wheel or slider zoom inspects local detail.
          </p>
        </div>

        <div class="flex flex-wrap gap-2.5">
          <StatusBadge>
            Type: {{ currentKind?.label }}
          </StatusBadge>
          <StatusBadge>
            Channels: {{ numberOfChannels || 'Pending' }}
          </StatusBadge>
        </div>
      </div>

      <div class="mt-5 rounded-[20px] border border-[#d6dfef] bg-white p-4 md:p-5">
        <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <FieldControl label="Category" class="max-w-xs">
            <select
              :value="previewCategory"
              class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
              @change="datasetStore.setPreviewCategory(($event.target as HTMLSelectElement).value)"
            >
              <option value="" disabled>Select category</option>
              <option
                v-for="option in categoryOptions"
                :key="option.key"
                :value="option.key"
              >
                {{ option.label }}
              </option>
            </select>
          </FieldControl>

          <div class="flex flex-wrap gap-2.5">
            <StatusBadge v-if="previewRowSource" tone="muted">
              Source: {{ previewRowSource }}
            </StatusBadge>
            <StatusBadge v-if="previewRowValues.length" tone="muted">
              Row Width: {{ previewRowValues.length }}
            </StatusBadge>
          </div>
        </div>

        <div
          v-if="isPreviewLoading || previewError || previewDataError"
          class="mt-5 flex min-h-80 items-center justify-center rounded-[16px] border border-dashed border-[#c9d7ea] bg-[#fafafa] px-6 text-center text-sm leading-[1.7] text-[#58708f]"
        >
          {{
            isPreviewLoading
              ? 'Loading one row from the selected category...'
              : previewError || previewDataError
          }}
        </div>
        <div v-else class="mt-5">
          <WaveformPreviewChart :channels="waveformChannels" />
        </div>
      </div>
    </SectionCard>

    <DialogModal :open="isChannelDialogOpen" @close="handleDialogCancel">
        <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Dataset Config</p>
        <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">
          Set Number of Channels
        </h3>
        <p class="mt-3 text-[15px] leading-[1.7] text-[#475a77]">
          Each row is one sample. Enter how many channels are interleaved inside the row,
          for example `x1 y1 z1 x2 y2 z2 ...` means `3` channels.
        </p>

        <FieldControl label="Number of Channels" class="mt-6">
          <input
            v-model="pendingNumberOfChannels"
            class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
            type="number"
            min="1"
            placeholder="For example: 3"
          />
        </FieldControl>

        <div class="mt-6 flex justify-end gap-3">
          <button
            type="button"
            class="rounded-xl border border-[#c9d7ea] px-4 py-2.5 text-sm font-semibold text-[#324869] transition hover:border-[#0d234a]"
            @click="handleDialogCancel"
          >
            Cancel
          </button>
          <button
            type="button"
            class="rounded-xl bg-[#0d234a] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#132f63] disabled:cursor-not-allowed disabled:bg-[#c9d7ea]"
            :disabled="!canConfirmChannels"
            @click="datasetStore.confirmChannelDialog()"
          >
            Confirm
          </button>
        </div>
    </DialogModal>
  </section>
</template>
