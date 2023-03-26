<script setup lang="ts">
import Input from './components/Input.vue'
import Screen from './components/Screen.vue'

import { reactive, ref } from 'vue'
import { io, Socket } from "socket.io-client";

import Loading from 'vue-loading-overlay';
import 'vue-loading-overlay/dist/css/index.css';

export interface Setting {
  mode:string;
  inputType:string;
  inputPath:string;
  device:string;
  emulatorName:string;
  port:number;
  fps_limit:number;
  bitrate:number;
  syncVideo:boolean;
  model_path:string;
  runtime:string;
  cropType:string;
  delay:number;
  saveFolder:string;
  interface:string;
  webport:number;
  grid:string;
  yolov8:boolean;
}

let setting = reactive<Setting>({
  mode: 'predict',
  inputType: 'adb',
  inputPath: '',
  device: 'bluestack',
  emulatorName: '',
  port: 5555,
  fps_limit: 60,
  bitrate: 16000000,
  syncVideo: true,
  model_path: '',
  runtime: 'onnx_runtime',
  cropType: 'grid',
  delay: 2000,
  saveFolder: '',
  interface: 'gui',
  webport: 8088,
  grid: '',
  yolov8: true
});

const isLoading = ref(import.meta.env.DEV ? false : true);
const start = ref(false);

interface ServerToClientEvents {
  init: (_setting:string) => void;
  finish: () => void;
}

interface ClientToServerEvents {
  run: (_setting:string) => void;
  stop: () => void;
}

const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io();

socket.on("init", (_setting:string) => {
  console.log(_setting);
  setting = reactive<Setting>(JSON.parse(_setting));
  isLoading.value = false;
});

socket.on("finish", () => {
  console.log('finish');
  start.value = false;
});

function run(){
  const send_str:string = JSON.stringify(setting);
  socket.emit("run", send_str);
  start.value = true;
}

function stop(){
  socket.emit("stop");
}

</script>

<template>
  <v-app>
    <v-main>
      <!-- <v-toolbar density="compact" color="blue-grey-lighten-5">
        <v-toolbar-title>Random Dice Roll Roll Tools</v-toolbar-title>
      </v-toolbar> -->

      <loading v-model:active="isLoading" is-full-page/>

      <v-container class="bg" fluid>
        <v-row no-gutters>
          <v-col>
            <Input v-model:setting="setting" :isLoading="isLoading"/>
          </v-col>
          <v-col cols="5">
            <Screen :start="start"/>
          </v-col>
        </v-row>
      </v-container>
    </v-main>
    <v-footer class="bg-light-green-lighten-3" border app rounded>
      <v-container fluid>
        <v-row align="center" justify="space-around" no-gutters>
          <v-col cols="10">
            <!-- <v-progress-linear
              model-value="20"
              color="light-green-darken-2"
              height="15"
              striped
            ></v-progress-linear> -->
          </v-col>
          <!-- <v-spacer></v-spacer> -->
          <v-col cols="1">
            <v-btn 
              prepend-icon="mdi-play" 
              size="large" 
              color="light-green-darken-2"
              v-if="!start"
              @click="run">
              Run
            </v-btn>
            <v-btn 
              prepend-icon="mdi-play" 
              size="large" 
              color="red-lighten-1"
              v-if="start"
              @click="stop">
              Stop
            </v-btn>
          </v-col>
        </v-row>
      </v-container>
    </v-footer>
  </v-app>
</template>

<style scoped>

.bg{
  background-color: #465362;
}

</style>
