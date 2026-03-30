<script setup lang="ts">
import { computed } from 'vue'
import DialogModal from '../../components/DialogModal.vue'
import FieldControl from '../../components/FieldControl.vue'
import InfoPlaceholder from '../../components/InfoPlaceholder.vue'
import SectionCard from '../../components/SectionCard.vue'
import StatusBadge from '../../components/StatusBadge.vue'
import { useDatasetClassificationStore } from '../../stores/datasetClassification'
import { useDatasetRegressionStore } from '../../stores/datasetRegression'
import { useOneDWorkflowStore } from '../../stores/oneDWorkflow'
import { useTrainingClassificationStore } from '../../stores/trainingClassification'

const workflowStore = useOneDWorkflowStore()
const classificationStore = useDatasetClassificationStore()
const regressionStore = useDatasetRegressionStore()
const classificationTrainingStore = useTrainingClassificationStore()

const isDatasetReady = computed(() =>
  workflowStore.activeKind === 'data-classification'
    ? classificationStore.isDatasetReady
    : regressionStore.isDatasetReady,
)

const isClassification = computed(() => workflowStore.activeKind === 'data-classification')
const classificationCategoryCount = computed(() => classificationStore.categoryOptions.length)
const shouldPromptSplit = computed(
  () => isClassification.value && classificationStore.datasetMode === 'raw-class-files',
)

function handleTrainClick() {
  if (shouldPromptSplit.value) {
    classificationTrainingStore.openSplitDialog()
    return
  }
}
</script>

<template>
  <section
    v-if="!isDatasetReady"
    class="flex min-h-[calc(100vh-170px)] items-center justify-center px-6 py-10"
  >
    <InfoPlaceholder
      eyebrow="Training"
      title="Waiting For Dataset Upload"
      description="Upload and confirm the dataset in the Dataset stage before configuring training."
      centered
      large
    />
  </section>

  <InfoPlaceholder
    v-else-if="!isClassification"
    eyebrow="Training"
    title="Regression Training"
    description="Regression training configuration will follow after the classification training page is finalized."
    :bullets="[
      'Keep using the current dataset readiness gate.',
      'Reuse the same training page shell once regression inputs are confirmed.',
    ]"
  />

  <section v-else class="flex min-h-full flex-col gap-5 p-6">
    <div class="grid gap-5 xl:grid-cols-[minmax(0,1.05fr)_minmax(340px,0.95fr)]">
      <SectionCard>
        <header class="mb-5">
          <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Training</p>
          <h2 class="m-0 text-[28px] font-bold leading-[1.2] text-[#0d234a]">Model</h2>
        </header>

        <div>
          <p class="m-0 text-sm font-semibold text-[#18325c]">Training Parameters</p>
          <div class="mt-4 grid gap-4 md:grid-cols-2">
            <FieldControl label="Epochs">
              <input
                v-model="classificationTrainingStore.epochs"
                class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
                type="number"
                min="1"
              />
            </FieldControl>
            <FieldControl label="Initial Learning Rate">
              <input
                v-model="classificationTrainingStore.learningRate"
                class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
                type="number"
                min="0.000001"
                step="0.0001"
              />
            </FieldControl>
            <FieldControl label="Batch Size">
              <input
                v-model="classificationTrainingStore.batchSize"
                class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
                type="number"
                min="1"
              />
            </FieldControl>
            <FieldControl label="Optimizer">
              <select
                v-model="classificationTrainingStore.optimizer"
                class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
              >
                <option value="adam">Adam</option>
                <option value="adamw">AdamW</option>
                <option value="sgd">SGD</option>
              </select>
            </FieldControl>
            <FieldControl label="Input Normalization">
              <select
                v-model="classificationTrainingStore.inputNormalization"
                class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
              >
                <option value="none">None</option>
                <option value="z_score">Z-Score</option>
                <option value="min_max">Min-Max</option>
              </select>
            </FieldControl>
          </div>
        </div>

        <div class="mt-6">
          <p class="m-0 text-sm font-semibold text-[#18325c]">Model Structure</p>
          <div class="mt-4 grid gap-4">
            <FieldControl label="Model Preset">
              <select
                v-model="classificationTrainingStore.modelPreset"
                class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
              >
                <option value="mlp">MLP</option>
                <option value="cnn">1D CNN</option>
              </select>
            </FieldControl>
          </div>
        </div>

        <div
          v-if="classificationTrainingStore.modelPreset === 'mlp'"
          class="mt-5 rounded-[20px] border border-[#d6dfef] bg-[#fbfcfe] p-5"
        >
          <div class="mb-4 flex items-center justify-between gap-3">
            <div>
              <p class="m-0 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">MLP Structure</p>
              <h3 class="m-0 mt-2 text-[22px] font-bold leading-[1.2] text-[#0d234a]">Editable Dense Layers</h3>
            </div>
            <StatusBadge tone="muted">Output: {{ classificationCategoryCount }} Classes</StatusBadge>
          </div>

          <div class="flex flex-col gap-3">
            <div
              v-for="(layer, index) in classificationTrainingStore.hiddenLayers"
              :key="layer.id"
              class="relative rounded-[18px] bg-white px-4 py-4"
              :class="layer.removable ? 'border border-dashed border-[#7b93b6]' : 'border border-[#d6dfef]'"
            >
              <button
                type="button"
                class="flex w-full items-center justify-center rounded-xl px-10 py-2 text-center text-sm font-semibold text-[#18325c] transition hover:bg-[#f7f9fc]"
                @click="classificationTrainingStore.openLayerDialog(layer.id)"
              >
                Dense Layer {{ index + 1 }} ({{ layer.units || '0' }} neurons)
              </button>
              <button
                v-if="layer.removable"
                type="button"
                class="absolute right-3 top-3 rounded-lg border border-[#f1b4bc] bg-[#fff4f6] px-3 py-1.5 text-xs font-semibold text-[#c24157] transition hover:border-[#dc5168] hover:bg-[#ffe8ec]"
                @click="classificationTrainingStore.removeDenseLayer(layer.id)"
              >
                Remove
              </button>
            </div>

            <button
              type="button"
              class="rounded-[18px] border border-dashed border-[#7b93b6] bg-[#f7f9fc] px-4 py-5 text-center transition hover:border-[#0d234a] hover:bg-[#eef3fb]"
              @click="classificationTrainingStore.addDenseLayer"
            >
              <p class="m-0 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Network Extension</p>
              <p class="m-0 mt-2 text-base font-semibold text-[#0d234a]">- - - Add Dense Layer - - -</p>
              <p class="mb-0 mt-2 text-sm leading-[1.7] text-[#58708f]">
                Insert a new hidden dense layer before the output layer.
              </p>
            </button>

            <div class="rounded-[18px] border border-[#d6dfef] bg-white px-4 py-4">
              <p class="m-0 text-center text-sm font-semibold text-[#18325c]">
                Output Layer ({{ classificationCategoryCount }} classes)
              </p>
            </div>
          </div>
        </div>

        <div
          v-else
          class="mt-5 rounded-[20px] border border-[#d6dfef] bg-[#fbfcfe] p-5"
        >
          <div class="mb-4 flex items-center justify-between gap-3">
            <div>
              <p class="m-0 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">1D CNN Structure</p>
              <h3 class="m-0 mt-2 text-[22px] font-bold leading-[1.2] text-[#0d234a]">Editable Sequential Layers</h3>
            </div>
            <div class="flex items-center gap-3">
              <StatusBadge tone="muted">Output: {{ classificationCategoryCount }} Classes</StatusBadge>
              <button
                type="button"
                class="rounded-xl border border-[#c9d7ea] px-3 py-2 text-xs font-semibold text-[#324869] transition hover:border-[#0d234a] hover:text-[#0d234a]"
                @click="classificationTrainingStore.restoreCnnPreset"
              >
                Restore Preset
              </button>
            </div>
          </div>

          <div class="flex flex-col gap-3">
            <div class="rounded-[18px] border border-[#d6dfef] bg-white px-4 py-4">
              <p class="m-0 text-center text-sm font-semibold text-[#18325c]">
                Input Layer ({{ classificationStore.numberOfChannels || '0' }} channels)
              </p>
            </div>

            <div
              v-for="(layer, index) in classificationTrainingStore.cnnLayers"
              :key="layer.id"
              class="relative rounded-[18px] bg-white px-4 py-4"
              :class="layer.removable ? 'border border-dashed border-[#7b93b6]' : 'border border-[#d6dfef]'"
            >
              <button
                v-if="layer.type === 'cnn-block'"
                type="button"
                class="flex w-full items-center justify-center rounded-xl px-10 py-2 text-center text-sm font-semibold text-[#18325c] transition hover:bg-[#f7f9fc]"
                @click="classificationTrainingStore.openCnnBlockDialog(layer.id)"
              >
                CNN Block {{ index + 1 }} ({{ layer.filters || '0' }} filters, kernel {{ layer.kernelSize || '0' }}, pool {{ layer.poolSize || '0' }})
              </button>
              <button
                v-else-if="layer.type === 'dense'"
                type="button"
                class="flex w-full items-center justify-center rounded-xl px-10 py-2 text-center text-sm font-semibold text-[#18325c] transition hover:bg-[#f7f9fc]"
                @click="classificationTrainingStore.openCnnDenseDialog(layer.id)"
              >
                Dense Layer ({{ layer.units || '0' }} neurons)
              </button>
              <button
                v-else-if="layer.type === 'dropout'"
                type="button"
                class="flex w-full items-center justify-center rounded-xl px-10 py-2 text-center text-sm font-semibold text-[#18325c] transition hover:bg-[#f7f9fc]"
                @click="classificationTrainingStore.openCnnDropoutDialog(layer.id)"
              >
                Dropout Layer (rate {{ layer.rate || '0' }})
              </button>
              <p
                v-else
                class="m-0 text-center text-sm font-semibold text-[#18325c]"
              >
                GlobalAveragePooling1D
              </p>
              <button
                v-if="layer.removable"
                type="button"
                class="absolute right-3 top-3 rounded-lg border border-[#f1b4bc] bg-[#fff4f6] px-3 py-1.5 text-xs font-semibold text-[#c24157] transition hover:border-[#dc5168] hover:bg-[#ffe8ec]"
                @click="classificationTrainingStore.removeCnnLayer(layer.id)"
              >
                Remove
              </button>
            </div>

            <div class="rounded-[18px] border border-dashed border-[#7b93b6] bg-[#f7f9fc] px-4 py-5 text-center">
              <p class="m-0 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Network Extension</p>
              <div class="mt-4 grid gap-3 md:grid-cols-3">
                <button
                  type="button"
                  class="rounded-xl border border-dashed border-[#7b93b6] bg-white px-4 py-3 text-sm font-semibold text-[#18325c] transition hover:border-[#0d234a] hover:bg-[#eef3fb]"
                  @click="classificationTrainingStore.addCnnBlock"
                >
                  Add CNN Block
                </button>
                <button
                  type="button"
                  class="rounded-xl border border-dashed border-[#7b93b6] bg-white px-4 py-3 text-sm font-semibold text-[#18325c] transition hover:border-[#0d234a] hover:bg-[#eef3fb]"
                  @click="classificationTrainingStore.addCnnDenseLayer"
                >
                  Add Dense
                </button>
                <button
                  type="button"
                  class="rounded-xl border border-dashed border-[#7b93b6] bg-white px-4 py-3 text-sm font-semibold text-[#18325c] transition hover:border-[#0d234a] hover:bg-[#eef3fb]"
                  @click="classificationTrainingStore.addCnnDropoutLayer"
                >
                  Add Dropout
                </button>
              </div>
              <p class="mb-0 mt-3 text-sm leading-[1.7] text-[#58708f]">
                Append new layers above the fixed output layer.
              </p>
            </div>

            <div class="rounded-[18px] border border-[#d6dfef] bg-white px-4 py-4">
              <p class="m-0 text-center text-sm font-semibold text-[#18325c]">
                Output Layer ({{ classificationCategoryCount }} classes)
              </p>
            </div>
          </div>
        </div>

        <div class="mt-6 flex justify-center">
          <button
            type="button"
            class="min-w-[160px] rounded-xl bg-[#0d234a] px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-[#132f63]"
            @click="handleTrainClick"
          >
            Train
          </button>
        </div>
      </SectionCard>

      <SectionCard class="self-start">
        <header class="mb-5">
          <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Logs</p>
          <h2 class="m-0 text-[28px] font-bold leading-[1.2] text-[#0d234a]">Training Logs</h2>
        </header>

        <div class="rounded-[20px] border border-[#d6dfef] bg-[#0b1730] p-4 text-[#dce8ff] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
          <div class="flex items-center justify-between gap-3 border-b border-white/10 pb-3">
            <p class="m-0 text-sm font-semibold">Console</p>
            <StatusBadge tone="muted">Idle</StatusBadge>
          </div>

          <div class="mt-4 flex h-[320px] flex-col gap-3 overflow-auto font-mono text-[13px] leading-[1.7]">
            <p class="m-0 text-[#8aa3c7]">[system] Waiting for training start.</p>
            <p class="m-0 text-[#8aa3c7]">[dataset] Classification dataset is ready.</p>
            <p class="m-0 text-[#8aa3c7]">[model] Default preset: {{ classificationTrainingStore.modelPreset.toUpperCase() }}</p>
            <p class="m-0 text-[#8aa3c7]">[info] Accuracy, loss, confusion matrix, and t-SNE panels will attach after the log-first layout is confirmed.</p>
          </div>
        </div>
      </SectionCard>
    </div>

    <DialogModal :open="classificationTrainingStore.isSplitDialogOpen" @close="classificationTrainingStore.closeSplitDialog">
      <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Train/Test Split</p>
      <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">Configure Split Before Training</h3>
      <p class="mt-3 text-[15px] leading-[1.7] text-[#475a77]">
        This dataset was uploaded as raw class files. Set the train/test split here before the training request is assembled.
      </p>

      <div class="mt-6 grid gap-4 md:grid-cols-3">
        <FieldControl label="Train Percent">
          <input
            v-model="classificationTrainingStore.splitTrainPercent"
            class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
            type="number"
            min="1"
            max="99"
          />
        </FieldControl>
        <FieldControl label="Test Percent">
          <input
            v-model="classificationTrainingStore.splitTestPercent"
            class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
            type="number"
            min="1"
            max="99"
          />
        </FieldControl>
        <FieldControl label="Shuffle Seed">
          <input
            v-model="classificationTrainingStore.splitSeed"
            class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
            type="number"
            min="0"
          />
        </FieldControl>
      </div>

      <div class="mt-5 rounded-[16px] border border-dashed border-[#c9d7ea] bg-[#fafafa] px-5 py-4 text-sm leading-[1.7] text-[#58708f]">
        {{ classificationTrainingStore.splitConfigError || 'Split configuration is valid. Continue will close the dialog in this front-end prototype.' }}
      </div>

      <div class="mt-6 flex justify-end gap-3">
        <button
          type="button"
          class="rounded-xl border border-[#c9d7ea] px-4 py-2.5 text-sm font-semibold text-[#324869] transition hover:border-[#0d234a]"
          @click="classificationTrainingStore.closeSplitDialog"
        >
          Cancel
        </button>
        <button
          type="button"
          class="rounded-xl bg-[#0d234a] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#132f63] disabled:cursor-not-allowed disabled:bg-[#c9d7ea]"
          :disabled="Boolean(classificationTrainingStore.splitConfigError)"
          @click="classificationTrainingStore.closeSplitDialog"
        >
          Continue
        </button>
      </div>
    </DialogModal>

    <DialogModal :open="classificationTrainingStore.isLayerDialogOpen" @close="classificationTrainingStore.closeLayerDialog">
      <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">Dense Layer</p>
      <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">Edit Neurons</h3>

      <FieldControl label="Neuron Count" class="mt-6">
        <input
          v-model="classificationTrainingStore.editingLayerUnits"
          class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
          type="number"
          min="1"
        />
      </FieldControl>

      <div class="mt-6 flex justify-end gap-3">
        <button
          type="button"
          class="rounded-xl border border-[#c9d7ea] px-4 py-2.5 text-sm font-semibold text-[#324869] transition hover:border-[#0d234a]"
          @click="classificationTrainingStore.closeLayerDialog"
        >
          Cancel
        </button>
        <button
          type="button"
          class="rounded-xl bg-[#0d234a] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#132f63]"
          @click="classificationTrainingStore.confirmLayerDialog"
        >
          Confirm
        </button>
      </div>
    </DialogModal>

    <DialogModal :open="classificationTrainingStore.isCnnBlockDialogOpen" @close="classificationTrainingStore.closeCnnBlockDialog">
      <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">CNN Block</p>
      <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">Edit Block Parameters</h3>

      <div class="mt-6 grid gap-4 md:grid-cols-3">
        <FieldControl label="Filters">
          <input
            v-model="classificationTrainingStore.editingCnnFilters"
            class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
            type="number"
            min="1"
          />
        </FieldControl>
        <FieldControl label="Kernel Size">
          <input
            v-model="classificationTrainingStore.editingCnnKernelSize"
            class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
            type="number"
            min="1"
          />
        </FieldControl>
        <FieldControl label="Pool Size">
          <input
            v-model="classificationTrainingStore.editingCnnPoolSize"
            class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
            type="number"
            min="1"
          />
        </FieldControl>
      </div>

      <div class="mt-6 flex justify-end gap-3">
        <button
          type="button"
          class="rounded-xl border border-[#c9d7ea] px-4 py-2.5 text-sm font-semibold text-[#324869] transition hover:border-[#0d234a]"
          @click="classificationTrainingStore.closeCnnBlockDialog"
        >
          Cancel
        </button>
        <button
          type="button"
          class="rounded-xl bg-[#0d234a] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#132f63]"
          @click="classificationTrainingStore.confirmCnnBlockDialog"
        >
          Confirm
        </button>
      </div>
    </DialogModal>

    <DialogModal :open="classificationTrainingStore.isCnnDenseDialogOpen" @close="classificationTrainingStore.closeCnnDenseDialog">
      <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">CNN Dense Layer</p>
      <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">Edit Dense Layer</h3>

      <FieldControl label="Neuron Count" class="mt-6">
        <input
          v-model="classificationTrainingStore.editingCnnDenseUnits"
          class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
          type="number"
          min="1"
        />
      </FieldControl>

      <div class="mt-6 flex justify-end gap-3">
        <button
          type="button"
          class="rounded-xl border border-[#c9d7ea] px-4 py-2.5 text-sm font-semibold text-[#324869] transition hover:border-[#0d234a]"
          @click="classificationTrainingStore.closeCnnDenseDialog"
        >
          Cancel
        </button>
        <button
          type="button"
          class="rounded-xl bg-[#0d234a] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#132f63]"
          @click="classificationTrainingStore.confirmCnnDenseDialog"
        >
          Confirm
        </button>
      </div>
    </DialogModal>

    <DialogModal :open="classificationTrainingStore.isCnnDropoutDialogOpen" @close="classificationTrainingStore.closeCnnDropoutDialog">
      <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">CNN Dropout Layer</p>
      <h3 class="m-0 text-[24px] font-bold leading-[1.2] text-[#0d234a]">Edit Dropout Layer</h3>

      <FieldControl label="Dropout Rate" class="mt-6">
        <input
          v-model="classificationTrainingStore.editingCnnDropoutRate"
          class="h-[42px] w-full rounded-xl border border-[#c9d7ea] bg-white px-3.5 text-sm text-[#18325c] outline-none transition focus:border-[#0d234a] focus:shadow-[0_0_0_3px_rgba(13,35,74,0.08)]"
          type="number"
          min="0"
          max="1"
          step="0.1"
        />
      </FieldControl>

      <div class="mt-6 flex justify-end gap-3">
        <button
          type="button"
          class="rounded-xl border border-[#c9d7ea] px-4 py-2.5 text-sm font-semibold text-[#324869] transition hover:border-[#0d234a]"
          @click="classificationTrainingStore.closeCnnDropoutDialog"
        >
          Cancel
        </button>
        <button
          type="button"
          class="rounded-xl bg-[#0d234a] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#132f63]"
          @click="classificationTrainingStore.confirmCnnDropoutDialog"
        >
          Confirm
        </button>
      </div>
    </DialogModal>
  </section>
</template>
