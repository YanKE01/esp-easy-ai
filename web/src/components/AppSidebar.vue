<script setup lang="ts">
import { useRoute } from 'vue-router'
import type { Component } from 'vue'

interface NavItem {
  label: string
  to: string
  icon?: Component
  activePrefix?: string
}

const props = defineProps<{
  items: NavItem[]
}>()

const route = useRoute()

function isItemActive(item: NavItem) {
  if (route.path === item.to) {
    return true
  }

  if (item.activePrefix) {
    return route.path.startsWith(item.activePrefix)
  }

  return false
}

function getItemStyle(item: NavItem) {
  if (!isItemActive(item)) {
    return undefined
  }

  return {
    backgroundColor: '#f9ce25',
    color: '#0d234a',
  }
}

</script>

<template>
  <aside class="sticky top-[50px] h-[calc(100vh-50px)] w-28 shrink-0 self-start overflow-y-auto bg-[#0d234a] px-0 py-2.5 shadow-[0_4px_16px_rgba(8,20,45,0.16)]">
    <nav class="flex flex-col items-center gap-2.5">
      <RouterLink
        v-for="item in props.items"
        :key="item.to"
        :to="item.to"
        class="flex min-h-[84px] w-[84px] flex-col items-center justify-center gap-2 rounded-xl bg-transparent px-2 py-3 text-center text-[#dce8ff] transition hover:bg-white/12"
        :style="getItemStyle(item)"
      >
        <component
          :is="item.icon"
          v-if="item.icon"
          :size="32"
          :stroke-width="2"
          class="shrink-0"
        />
        <span class="text-[12px] font-medium leading-3 tracking-[0.1em]">
          {{ item.label }}
        </span>
      </RouterLink>
    </nav>
  </aside>
</template>
