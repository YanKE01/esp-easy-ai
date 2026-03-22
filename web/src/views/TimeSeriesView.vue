<script setup lang="ts">
import { ref } from 'vue'
import DatasetTab from './time-series/DatasetTab.vue'
import TrainingTab from './time-series/TrainingTab.vue'
import EmulationTab from './time-series/EmulationTab.vue'

const tabs = [
  { id: 'dataset', label: 'Dataset' },
  { id: 'training', label: 'Training' },
  { id: 'emulation', label: 'Emulation' },
]

const activeTab = ref('dataset')
</script>

<template>
  <div class="page-view">
    <div class="top-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        class="top-tab"
        :class="{ 'is-active': activeTab === tab.id }"
        type="button"
        @click="activeTab = tab.id"
      >
        {{ tab.label }}
      </button>
    </div>

    <div class="tab-content">
      <DatasetTab v-show="activeTab === 'dataset'" />
      <TrainingTab v-show="activeTab === 'training'" />
      <EmulationTab v-show="activeTab === 'emulation'" />
    </div>
  </div>
</template>
