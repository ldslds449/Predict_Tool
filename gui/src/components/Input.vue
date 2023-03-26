<script setup lang="ts">
import { Setting } from '../App.vue'
import { watch, PropType } from 'vue'

const props = defineProps({
  setting: {
    type: Object as PropType<Setting>,
    required: true
  },
  isLoading:{
    type: Boolean,
    required: true
  }
});
const emit = defineEmits(['update:setting']);

watch(props.setting, (selection, prevSelection) => { 
  if(props.isLoading === false){
    emit('update:setting', props.setting);
  }
});

const rules = {
  required: (value:string):boolean|string => { return value.toString().length > 0 || 'Required.'; },
  port: (value:number):boolean|string => { return (Number(value) >= 1024 && Number(value) <= 65535) || 'Invalid port number.'; },
  fps: (value:number):boolean|string => { return (Number(value) >= 1 && Number(value) <= 300) || 'Invalid number.'; },
  bitrate: (value:number):boolean|string => { return (Number(value) >= 1000000 && Number(value) <= 100000000) || 'Invalid number.'; },
  delay: (value:number):boolean|string => { return (Number(value) >= 0 && Number(value) <= 60000) || 'Invalid number.'; },
}

</script>

<template>
  <v-container fuild>
    <v-sheet class="pa-6" color="blue-grey-lighten-5" :rounded=true>
      <!-- mode -->
      <v-row align="center" no-gutters class="mb-6">
        <v-col cols="3">
          <p class="text-left">
            Mode
          </p>
        </v-col>
        <v-col>
          <v-radio-group
            v-model="props.setting.mode"
            inline
            hide-details
          >
            <v-radio
              label="Predict"
              value="predict"
            ></v-radio>
            <v-radio
              label="Crop"
              value="crop"
            ></v-radio>
            <v-radio
              label="Record"
              value="record"
            ></v-radio>
          </v-radio-group>
        </v-col>
      </v-row>
      
      <!-- input -->
      <v-row align="center" no-gutters class="my-4">
        <v-col cols="3">
          <p class="text-left">
            Input Type
          </p>
        </v-col>
        <v-col>
          <div>
            <v-radio-group
              v-model="props.setting.inputType"
              inline
              hide-details
            >
              <v-radio
                label="ADB"
                value="adb"
              ></v-radio>
              <v-radio
                label="Video"
                value="video"
              ></v-radio>
              <v-radio
                label="Image"
                value="image"
              ></v-radio>
            </v-radio-group>
          </div>
        </v-col>
      </v-row>

      <!-- input path -->
      <v-row align="center" no-gutters>
        <v-col cols="3">
          <p class="text-left">
            {{ props.setting.inputType === 'adb' ? 'Port' : 'Input Path' }}
          </p>
        </v-col>
        <v-col class="px-2">
          <v-text-field 
            :prepend-icon='props.setting.inputType === "image" ? "mdi-image" : "mdi-video"'
            label="File" 
            variant="underlined" 
            v-model="props.setting.inputPath"
            :placeholder='props.setting.inputType === "image" ? "data/images/img.png" : "data/videos/video.mp4"'
            v-if="props.setting.inputType !== 'adb'"
            :rules="[rules.required]">
          </v-text-field>
          <v-text-field 
            prepend-icon="mdi-anchor"
            label="Port" 
            variant="underlined"
            placeholder="5555"
            v-model.number="props.setting.port"
            v-if="props.setting.inputType === 'adb'"
            :rules="[rules.port]">
          </v-text-field>
        </v-col>
      </v-row>

      <!-- device -->
      <v-row align="center" no-gutters class="my-4" v-if="props.setting.inputType === 'adb'">
        <v-col cols="3">
          <p class="text-left">
            Device
          </p>
        </v-col>
        <v-col>
          <div>
            <v-radio-group
              v-model="props.setting.device"
              inline
              hide-details
            >
              <v-radio
                label="BlueStack"
                value="bluestack"
              ></v-radio>
              <v-radio
                label="Other"
                value="other"
              ></v-radio>
            </v-radio-group>
          </div>
        </v-col>
      </v-row>

      <!-- emulator name -->
      <v-row align="center" no-gutters v-if="props.setting.inputType === 'adb'">
        <v-col cols="3">
          <p class="text-left">
            Emulator Name
          </p>
        </v-col>
        <v-col class="px-2">
          <v-text-field 
            prepend-icon="mdi-tag"
            label="Emulator Name" 
            variant="underlined"
            placeholder="BlueStacks App Player"
            v-model="props.setting.emulatorName"
            :rules="[rules.required]">
          </v-text-field>
        </v-col>
      </v-row>

      <!-- FPS Limit -->
      <v-row align="center" no-gutters v-if="props.setting.inputType === 'adb'">
        <v-col cols="3">
          <p class="text-left">
            FPS Limit
          </p>
        </v-col>
        <v-col class="px-2">
          <v-text-field 
            prepend-icon="mdi-animation-outline"
            label="FPS Limit" 
            variant="underlined"
            placeholder="60"
            v-model.number="props.setting.fps_limit"
            :rules="[rules.fps]">
          </v-text-field>
        </v-col>
      </v-row>

      <!-- bitrate -->
      <v-row align="center" no-gutters v-if="props.setting.inputType === 'adb'">
        <v-col cols="3">
          <p class="text-left">
            Bitrate
          </p>
        </v-col>
        <v-col class="px-2">
          <v-text-field 
            prepend-icon="mdi-chip"
            label="Bitrate" 
            variant="underlined"
            placeholder="16000000"
            v-model.number="props.setting.bitrate"
            :rules="[rules.bitrate]">
          </v-text-field>
        </v-col>
      </v-row>

      <div v-if="props.setting.mode === 'predict'">
        <v-list-subheader>
          Predict
        </v-list-subheader>
        <v-divider :thickness="4" class="mb-4"></v-divider>

        <!-- sync video -->
        <v-row align="center" no-gutters v-if="props.setting.inputType === 'video'" class="mb-3">
          <v-col cols="3">
            <p class="text-left">
              Sync Video
            </p>
          </v-col>
          <v-col>
            <v-checkbox
              v-model="props.setting.syncVideo"
              label="Sync Video"
              :value="true"
              hide-details
            ></v-checkbox>
          </v-col>
        </v-row>
      </div>

      <div v-if="props.setting.mode === 'crop'">
        <v-list-subheader>
          Crop
        </v-list-subheader>
        <v-divider :thickness="4" class="mb-6"></v-divider>

        <!-- crop type -->
        <v-row align="center" no-gutters class="my-4">
          <v-col cols="3">
            <p class="text-left">
              Type
            </p>
          </v-col>
          <v-col>
            <v-radio-group
              v-model="props.setting.cropType"
              inline
              hide-details
            >
              <v-radio
                label="Grid"
                value="grid"
              ></v-radio>
              <v-radio
                label="Dian"
                value="dian"
              ></v-radio>
            </v-radio-group>
          </v-col>
        </v-row>

        <!-- save folder -->
        <v-row align="center" no-gutters>
          <v-col cols="3">
            <p class="text-left">
              Save Folder
            </p>
          </v-col>
          <v-col class="px-2">
            <v-text-field 
              prepend-icon="mdi-folder"
              label="Save Folder" 
              variant="underlined"
              placeholder="data/videos"
              v-model="props.setting.saveFolder"
              :rules="[rules.required]">
            </v-text-field>
          </v-col>
        </v-row>

        <!-- delay -->
        <v-row align="center" no-gutters v-if="props.setting.inputType === 'adb' || props.setting.inputType === 'video'">
          <v-col cols="3">
            <p class="text-left">
              Delay
            </p>
          </v-col>
          <v-col class="px-2">
            <v-text-field 
              prepend-icon="mdi-clock-time-eight-outline"
              label="Delay" 
              variant="underlined"
              placeholder="2000"
              v-model.number="props.setting.delay"
              :rules="[rules.delay]">
            </v-text-field>
          </v-col>
        </v-row>
      </div>
    
      <div v-if="props.setting.mode === 'record'">
        <v-list-subheader>
          Record
        </v-list-subheader>
        <v-divider :thickness="4" class="mb-6"></v-divider>

        <!-- save folder -->
        <v-row align="center" no-gutters>
          <v-col cols="3">
            <p class="text-left">
              Save Folder
            </p>
          </v-col>
          <v-col class="px-2">
            <v-text-field 
              prepend-icon="mdi-folder"
              label="Save Folder" 
              variant="underlined"
              placeholder="data/videos"
              v-model="props.setting.saveFolder"
              :rules="[rules.required]">
            </v-text-field>
          </v-col>
        </v-row>
      </div>

      <div v-if="(props.setting.mode === 'crop' && props.setting.cropType === 'dian') || props.setting.mode === 'predict'">
        <!-- model -->
        <v-row align="center" no-gutters>
          <v-col cols="3">
            <p class="text-left">
              Model
            </p>
          </v-col>
          <v-col class="px-2">
            <v-text-field 
              prepend-icon="mdi-star-david"
              label="Model" 
              variant="underlined"
              placeholder="data/models/best.onnx"
              v-model="props.setting.model_path"
              :rules="[rules.required]">
            </v-text-field>
          </v-col>
        </v-row>

        <!-- runtime -->
        <v-row align="center" no-gutters class="my-4">
          <v-col cols="3">
            <p class="text-left">
              Runtime
            </p>
          </v-col>
          <v-col>
            <v-radio-group
              v-model="props.setting.runtime"
              inline
              hide-details
            >
              <v-radio
                label="Opencv DNN"
                value="opencv_dnn"
              ></v-radio>
              <v-radio
                label="ONNX Runtime"
                value="onnx_runtime"
              ></v-radio>
              <v-radio
                label="Openvino"
                value="openvino"
              ></v-radio>
              <v-radio
                label="DeepSparse"
                value="deepsparse"
              ></v-radio>
            </v-radio-group>
          </v-col>
        </v-row>

        <!-- yolov8 -->
        <v-row align="center" no-gutters class="mb-3">
          <v-col cols="3">
            <p class="text-left">
              Yolov8
            </p>
          </v-col>
          <v-col>
            <v-checkbox
              v-model="props.setting.yolov8"
              label="Yolov8"
              :value="true"
              hide-details
            ></v-checkbox>
          </v-col>
        </v-row>
      </div>
      
      <!-- can not be modify -->
      <!-- <div v-if="props.setting.interface === 'web'">
        <v-list-subheader>
          Interface
        </v-list-subheader>
        <v-divider :thickness="4" class="mb-6"></v-divider>

        interface type
        <v-row align="center" no-gutters class="mb-6">
          <v-col cols="3">
            <p class="text-left">
              Interface
            </p>
          </v-col>
          <v-col>
            <v-radio-group
              v-model="props.setting.interface"
              inline
              hide-details
            >
              <v-radio
                label="CLI"
                value="cli"
              ></v-radio>
              <v-radio
                label="GUI"
                value="gui"
              ></v-radio>
              <v-radio
                label="Web"
                value="web"
              ></v-radio>
            </v-radio-group>
          </v-col>
        </v-row>

        web port
        <v-row align="center" no-gutters>
          <v-col cols="3">
            <p class="text-left">
              Web Port
            </p>
          </v-col>
          <v-col class="px-2">
            <v-text-field 
              prepend-icon="mdi-web"
              label="Web Port" 
              variant="underlined"
              placeholder="8080"
              v-model.number="props.setting.webport"
              :rules="[rules.port]">
            </v-text-field>
          </v-col>
        </v-row>
      </div> -->
      
      <v-list-subheader class="mt-4">
        Other
      </v-list-subheader>
      <v-divider :thickness="4" class="mb-6"></v-divider>
      
      <!-- Grid -->
      <v-row align="center" no-gutters>
        <v-col cols="3">
          <p class="text-left">
            Grid
          </p>
        </v-col>
        <v-col class="px-2">
          <v-textarea 
            label="Grid" 
            variant="outlined" 
            :model-value="props.setting.grid"
            auto-grow >
          </v-textarea>
        </v-col>
      </v-row>
    </v-sheet>
  </v-container>
</template>
