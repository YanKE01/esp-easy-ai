<script setup lang="ts">
defineProps<{
  label: string
  text: string
  inputId: string
}>()
</script>

<template>
  <label
    :for="inputId"
    class="mt-5 flex cursor-pointer flex-col rounded-2xl border border-dashed border-[#c9d7ea] bg-white px-5 py-6 transition hover:border-[#0d234a]"
  >
    <span class="text-sm font-semibold text-[#0d234a]">{{ label }}</span>
    <span class="mt-3 text-sm leading-[1.7] text-[#58708f]">{{ text }}</span>
  </label>
</template>
