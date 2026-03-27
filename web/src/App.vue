<script setup lang="ts">
import AppSidebar from './components/AppSidebar.vue'
import TopBar from './components/TopBar.vue'

import { ArrowUp10, SendToBack, Images as ImagesIcon } from '@lucide/vue'

const navItems = [
  { label: 'Overview', to: '/overview', icon: SendToBack },
  { label: '1-D', to: '/one-d/dataset', activePrefix: '/one-d/', icon: ArrowUp10 },
  { label: 'Image', to: '/image', icon: ImagesIcon },
]
</script>

<template>
  <div class="min-h-screen bg-[radial-gradient(circle_at_top_right,rgba(34,89,170,0.1),transparent_24%),linear-gradient(180deg,#eef3fb_0%,#f7f9fc_100%)] text-[#13233f]">
    <TopBar />

    <div class="flex min-h-[calc(100vh-50px)] pt-0">
      <AppSidebar :items="navItems" />

      <main class="min-w-0 flex-1 bg-[#fafafa] p-0">
        <RouterView />
      </main>
    </div>
  </div>
</template>
