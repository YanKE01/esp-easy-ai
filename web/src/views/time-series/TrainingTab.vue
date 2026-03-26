<script setup lang="ts">
import InfoPlaceholder from '../../components/InfoPlaceholder.vue'
</script>

<template>
  <InfoPlaceholder
    eyebrow="Training"
    title="Training Stage"
    description="The routed training view is in place. The next step should consume the dataset store rather than re-asking for data shape, channel count, or source layout."
    :bullets="[
      'Read dataset mode directly from Pinia.',
      'Bind training configuration to the current /one-d/training route.',
      'Avoid duplicating dataset upload or shape inference in this screen.',
    ]"
  />
</template>
