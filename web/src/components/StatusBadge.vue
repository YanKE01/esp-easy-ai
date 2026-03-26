<script setup lang="ts">
defineProps<{
  tone?: 'default' | 'muted'
}>()
</script>

<template>
  <span
    class="rounded-full px-4 py-2 font-semibold"
    :class="
      tone === 'muted'
        ? 'bg-[#eef3fb] text-xs text-[#324869]'
        : 'bg-[#eef3fb] text-sm text-[#0d234a]'
    "
  >
    <slot />
  </span>
</template>
