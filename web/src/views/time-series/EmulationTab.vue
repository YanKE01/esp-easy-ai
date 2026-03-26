<script setup lang="ts">
import InfoPlaceholder from '../../components/InfoPlaceholder.vue'
</script>

<template>
  <InfoPlaceholder
    eyebrow="Emulation"
    title="Emulation Stage"
    description="The emulation route is isolated and ready for later model output inspection, benchmark display, and device-target validation."
    :bullets="[
      'Consume the same global dataset assumptions as training.',
      'Use the route as the canonical entry for emulator-specific controls.',
      'Keep visual diagnostics and deployment checks scoped to this stage.',
    ]"
  />
</template>
