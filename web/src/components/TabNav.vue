<script setup lang="ts">
interface TabItem {
  label: string
  to: string
  disabled?: boolean
}

defineProps<{
  items: TabItem[]
}>()
</script>

<template>
  <nav class="flex gap-10 border-b border-[#e5e7eb] bg-[#fafafa] px-6 pt-3">
    <component
      v-for="item in items"
      :key="item.to"
      :is="item.disabled ? 'span' : 'RouterLink'"
      v-bind="item.disabled ? {} : { to: item.to }"
      class="relative border-0 bg-transparent px-0 pb-2.5 text-base font-medium tracking-[0.02em] text-[#6b7280] transition hover:text-[#0d234a]"
      :class="
        item.disabled
          ? 'cursor-not-allowed text-[#b8c1cf] hover:text-[#b8c1cf]'
          : ''
      "
      active-class="font-semibold text-[#0d234a] after:absolute after:bottom-[-1px] after:left-0 after:right-0 after:h-0.5 after:rounded-t-full after:bg-[#0d234a] after:content-['']"
    >
      {{ item.label }}
    </component>
  </nav>
</template>
