<script setup lang="ts">
import * as echarts from 'echarts'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

interface ChannelSeries {
  id: number
  label: string
  color: string
  values: number[]
}

interface Props {
  channels: ChannelSeries[]
}

const props = defineProps<Props>()

const chartRef = ref<HTMLElement | null>(null)
let chart: echarts.ECharts | null = null

const hasData = computed(() => props.channels.some((channel) => channel.values.length > 0))

function buildOption(): echarts.EChartsOption {
  const maxLength = Math.max(...props.channels.map((channel) => channel.values.length), 0)
  const xValues = Array.from({ length: maxLength }, (_, index) => index)

  return {
    animation: false,
    grid: {
      top: 56,
      right: 24,
      bottom: 72,
      left: 72,
    },
    legend: {
      top: 14,
      icon: 'roundRect',
      itemWidth: 14,
      itemHeight: 8,
      textStyle: {
        color: '#324869',
        fontSize: 12,
        fontWeight: 600,
      },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.96)',
      borderColor: '#d6dfef',
      borderWidth: 1,
      textStyle: {
        color: '#18325c',
      },
      axisPointer: {
        type: 'line',
        snap: true,
        lineStyle: {
          color: 'rgba(13, 35, 74, 0.65)',
          type: 'dashed',
          width: 1,
        },
      },
      formatter: (params: unknown) => {
        const items = Array.isArray(params) ? (params as any[]) : []

        if (!items.length) {
          return ''
        }

        const header = `<div style="font-weight:700;margin-bottom:6px;">Index ${items[0].axisValue}</div>`
        const rows = items
          .map((item) => {
            const value = Array.isArray(item.data) ? item.data[1] : item.data
            return `<div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
              <span style="display:inline-block;width:8px;height:8px;border-radius:999px;background:${item.color};"></span>
              <span>${item.seriesName}: <strong>${Number(value).toFixed(0)}</strong></span>
            </div>`
          })
          .join('')

        return `${header}${rows}`
      },
    },
    xAxis: {
      type: 'value',
      min: 0,
      max: Math.max(0, maxLength - 1),
      name: 'Sample Index',
      nameLocation: 'middle',
      nameGap: 42,
      axisLine: {
        lineStyle: {
          color: '#47607f',
        },
      },
      axisTick: {
        lineStyle: {
          color: '#47607f',
        },
      },
      axisLabel: {
        color: '#5a708e',
        fontSize: 11,
      },
      splitLine: {
        show: true,
        lineStyle: {
          color: 'rgba(88, 112, 143, 0.12)',
        },
      },
    },
    yAxis: {
      type: 'value',
      name: 'Value',
      nameLocation: 'middle',
      nameGap: 52,
      axisLine: {
        lineStyle: {
          color: '#47607f',
        },
      },
      axisTick: {
        lineStyle: {
          color: '#47607f',
        },
      },
      axisLabel: {
        color: '#5a708e',
        fontSize: 11,
      },
      splitLine: {
        show: true,
        lineStyle: {
          color: 'rgba(88, 112, 143, 0.14)',
        },
      },
    },
    dataZoom: [
      {
        type: 'inside',
        filterMode: 'none',
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        moveOnMouseWheel: false,
      },
      {
        type: 'slider',
        height: 26,
        bottom: 22,
        borderColor: '#d6dfef',
        brushSelect: false,
        fillerColor: 'rgba(37, 115, 214, 0.14)',
        backgroundColor: 'rgba(247, 250, 255, 0.95)',
        dataBackground: {
          lineStyle: {
            color: '#b8c9df',
          },
          areaStyle: {
            color: 'rgba(184, 201, 223, 0.24)',
          },
        },
      },
    ],
    series: props.channels.map((channel) => ({
      name: channel.label,
      type: 'line',
      smooth: false,
      showSymbol: false,
      symbol: 'circle',
      sampling: 'lttb',
      emphasis: {
        focus: 'series',
      },
      lineStyle: {
        width: 2.2,
        color: channel.color,
      },
      itemStyle: {
        color: channel.color,
      },
      data: xValues.map((xValue, index) => [xValue, channel.values[index]]),
    })),
  }
}

function renderChart() {
  if (!chartRef.value) {
    return
  }

  if (!chart) {
    chart = echarts.init(chartRef.value)
  }

  chart.setOption(buildOption(), true)
}

function handleResize() {
  chart?.resize()
}

onMounted(() => {
  if (!chartRef.value) {
    return
  }

  renderChart()
  window.addEventListener('resize', handleResize)
})

watch(
  () => props.channels,
  () => {
    renderChart()
  },
  { deep: true },
)

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  chart?.dispose()
  chart = null
})
</script>

<template>
  <div class="waveform-echart">
    <div v-if="!hasData" class="waveform-echart-empty">
      Select one category to preview a single row of data.
    </div>
    <div v-else ref="chartRef" class="waveform-echart-canvas"></div>
  </div>
</template>
