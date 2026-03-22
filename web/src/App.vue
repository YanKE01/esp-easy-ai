<script setup lang="ts">
import { computed, ref } from 'vue'
import TopBar from './components/TopBar.vue'
import Sidebar from './components/Sidebar.vue'
import InfoPlaceholder from './components/InfoPlaceholder.vue'
import TimeSeriesView from './views/TimeSeriesView.vue'

const navItems = [
  { label: 'Overview' },
  { label: '1D' },
  { label: 'Image' },
]

const activeItem = ref('Overview')

const placeholderViews = {
  Overview: {
    eyebrow: 'Project Scope',
    title: 'ESP EASY AI Workbench',
    description:
      'Build edge AI workflows by data shape. Start from the 1D workspace, then extend the same product line to image tasks later.',
    centered: true,
    large: true,
    bullets: [
      '1D is the active workbench for dataset setup, training, and emulation',
      'Dataset should define the data type so later steps inherit the same assumption',
      'Image remains a separate lane for future 2D workflows',
    ],
  },
  Image: {
    eyebrow: 'Image Workspace',
    title: 'Image Pipeline',
    description:
      'This area is reserved for the later image workflow after the 1D training path is settled.',
    bullets: [
      'Planned for image-oriented datasets and training tasks',
      'Kept separate from 1D so input shape and workflow assumptions stay clean',
    ],
  },
}

const currentView = computed(() => {
  if (activeItem.value === '1D') {
    return {
      component: TimeSeriesView,
      props: {},
    }
  }

  return {
    component: InfoPlaceholder,
    props: placeholderViews[activeItem.value as keyof typeof placeholderViews],
  }
})
</script>

<template>
  <div class="app-shell">
    <TopBar />

    <div class="workspace">
      <Sidebar
        :items="navItems"
        :active-item="activeItem"
        @select="activeItem = $event"
      />

      <main class="content">
        <component :is="currentView.component" v-bind="currentView.props" />
      </main>
    </div>
  </div>
</template>
