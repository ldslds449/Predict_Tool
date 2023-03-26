<script setup lang="ts">
import img_default from '@/assets/default.png'

const props = defineProps({
  start: {
    type: Boolean,
    required: true
  }
});

</script>

<template>
  <v-container>
    <v-sheet class="pa-2" align="center" color="blue-grey-lighten-5" :rounded=true>
      <v-img
        v-if="!start"
        :src="img_default">
      </v-img>
      <v-img
        v-if="start"
        :src="'' + '/video_feed'">
      </v-img>
    </v-sheet>
  </v-container>
</template>
