<script setup lang="ts">
import InfoPlaceholder from '../components/InfoPlaceholder.vue'
</script>

<template>
  <InfoPlaceholder
    eyebrow="Image Workspace"
    title="Image Pipeline"
    description="This area remains reserved for future image-oriented datasets and training flows. It stays separate from 1D so the routing and domain assumptions remain clean."
    :bullets="[
      'Planned for image-oriented datasets and training tasks.',
      'Separated from 1D to avoid mixing sequence and image pipeline assumptions.',
    ]"
  />
</template>
