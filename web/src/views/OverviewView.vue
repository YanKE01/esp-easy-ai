<script setup lang="ts">
import InfoPlaceholder from '../components/InfoPlaceholder.vue'
</script>

<template>
  <InfoPlaceholder
    eyebrow="Project Scope"
    title="ESP EASY AI Workbench"
    description="Build edge AI workflows by data shape. Start from the 1D workspace, then extend the same product line to image tasks later."
    centered
    large
    :bullets="[
      '1D is the active workbench for dataset setup, training, and emulation.',
      'Dataset type should be fixed early so later steps inherit the same assumptions.',
      'Image remains isolated so 2D workflows can evolve without leaking constraints into 1D.',
    ]"
  />
</template>
