<script setup lang="ts">
interface Props {
  eyebrow: string
  title: string
  description: string
  bullets?: string[]
  centered?: boolean
  large?: boolean
}

defineProps<Props>()
</script>

<template>
  <section class="placeholder-view" :class="{ 'is-centered': centered }">
    <div class="placeholder-card" :class="{ 'is-large': large }">
      <p class="placeholder-eyebrow">{{ eyebrow }}</p>
      <h2 class="placeholder-title">{{ title }}</h2>
      <p class="placeholder-description">{{ description }}</p>

      <ul v-if="bullets?.length" class="placeholder-list">
        <li v-for="item in bullets" :key="item">
          {{ item }}
        </li>
      </ul>
    </div>
  </section>
</template>
