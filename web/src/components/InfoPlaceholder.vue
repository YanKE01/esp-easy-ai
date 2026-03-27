<script setup lang="ts">
defineProps<{
  eyebrow: string
  title: string
  description: string
  bullets?: string[]
  centered?: boolean
  large?: boolean
}>()
</script>

<template>
  <section class="flex min-h-full p-6" :class="centered ? 'items-center justify-center' : 'items-start'">
    <div
      class="w-full rounded-[20px] border border-[rgba(13,35,74,0.08)] bg-[rgba(255,255,255,0.92)] shadow-[0_14px_32px_rgba(13,35,74,0.06)]"
      :class="large ? 'max-w-5xl p-10' : 'max-w-3xl p-8'"
    >
      <p class="m-0 mb-2 text-xs font-bold uppercase tracking-[0.08em] text-[#5e7498]">{{ eyebrow }}</p>
      <h2 class="m-0 text-[28px] font-bold leading-[1.2] text-[#0d234a]">{{ title }}</h2>
      <p class="mt-3 max-w-[700px] text-[15px] leading-[1.7] text-[#475a77]">{{ description }}</p>

      <ul v-if="bullets?.length" class="mt-6 list-disc space-y-2 pl-5 text-[14px] leading-[1.7] text-[#58708f]">
        <li
          v-for="item in bullets"
          :key="item"
        >
          {{ item }}
        </li>
      </ul>
    </div>
  </section>
</template>
