import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/overview',
    },
    {
      path: '/overview',
      name: 'overview',
      component: () => import('../views/OverviewView.vue'),
    },
    {
      path: '/one-d',
      component: () => import('../views/OneDLayout.vue'),
      children: [
        {
          path: '',
          redirect: '/one-d/dataset',
        },
        {
          path: 'dataset',
          name: 'dataset',
          component: () => import('../views/time-series/DatasetTab.vue'),
        },
        {
          path: 'training',
          name: 'training',
          component: () => import('../views/time-series/TrainingTab.vue'),
        },
        {
          path: 'emulation',
          name: 'emulation',
          component: () => import('../views/time-series/EmulationTab.vue'),
        },
      ],
    },
    {
      path: '/image',
      name: 'image',
      component: () => import('../views/ImageView.vue'),
    },
  ],
})

export default router
