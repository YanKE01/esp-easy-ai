<script setup lang="ts">
defineProps<{
  label: string
}>()
</script>

<template>
  <label class="block">
    <span class="mb-2 block text-sm font-semibold text-[#324869]">{{ label }}</span>
    <slot />
  </label>
</template>
