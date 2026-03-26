<script setup lang="ts">
defineProps<{
  open: boolean
}>()

defineEmits<{
  (e: 'close'): void
}>()
</script>

<template>
  <div
    v-if="open"
    class="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(13,35,74,0.28)] px-4"
    @click="$emit('close')"
  >
    <div
      class="w-full max-w-lg rounded-[20px] bg-white p-7 shadow-[0_20px_48px_rgba(13,35,74,0.14)]"
      @click.stop
    >
      <slot />
    </div>
  </div>
</template>
