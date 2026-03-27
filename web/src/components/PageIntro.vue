<script setup lang="ts">
defineProps<{
  eyebrow: string
  title: string
  description: string
}>()
</script>

<template>
  <header class="mb-5">
    <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">{{ eyebrow }}</p>
    <h2 class="m-0 text-[28px] font-bold leading-[1.2] text-[#0d234a]">
      {{ title }}
    </h2>
    <p class="mt-3 max-w-[700px] text-[15px] leading-[1.7] text-[#475a77]">
      {{ description }}
    </p>
  </header>
</template>
