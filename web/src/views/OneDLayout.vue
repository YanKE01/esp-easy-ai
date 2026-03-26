<script setup lang="ts">
import TabNav from '../components/TabNav.vue'

const tabs = [
  { label: 'Dataset', to: '/one-d/dataset' },
  { label: 'Training', to: '/one-d/training' },
  { label: 'Emulation', to: '/one-d/emulation' },
]
</script>

<template>
  <section class="flex min-h-full flex-col">
    <TabNav :items="tabs" />
    <div class="flex-1">
      <RouterView />
    </div>
  </section>
</template>
