<script setup lang="ts">
interface NavItem {
  label: string
}

interface Props {
  items: NavItem[]
  activeItem: string
}

defineProps<Props>()

defineEmits<{
  (e: 'select', value: string): void
}>()
</script>

<template>
  <aside class="sidebar">
    <nav class="sidebar-nav">
      <button
        v-for="item in items"
        :key="item.label"
        class="nav-tile"
        :class="{ 'is-active': activeItem === item.label }"
        type="button"
        :title="item.label"
        @click="$emit('select', item.label)"
      >
        <span class="nav-label">{{ item.label }}</span>
      </button>
    </nav>
  </aside>
</template>
