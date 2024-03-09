<script>
    import axios from 'axios'; // Make sure to install axios using npm install axios
    import client from '../client.js';
    import '../../app.css';
    import Carousel from 'svelte-carousel';
    


    let audioFile = null;
    let audioSegments = [];
    let audioCount = 0;
    let audioSrc = null;
    let wavePlotImages = [];
    let audioSegmentedRegions = [];
    let originalWavForm = null;
    let classification = Array(audioSegments.length).fill('');
    let genderDetection = Array(audioSegments.length).fill('');
    let genderPercentage = Array(audioSegments.length).fill('');
    let emotionDetection = Array(audioSegments.length).fill('');
    let forensicDetection = Array(audioSegments.length).fill('');

    function handleFileChange(event) {
        audioSrc = null;
        audioSegments = [];
        audioFile = event.target.files[0];
        console.log('File selected:', audioFile.name);
        audioSrc = URL.createObjectURL(audioFile);
    }

        function handleUpload() {
            if (!audioFile) {
                console.log('No file selected');
                return;
            }

            // Create a FormData object
            var formdata = new FormData();
            formdata.append("file", audioFile, audioFile.name);
            formdata.append("taken_at", new Date().toISOString())
            
            client.AudioAnalysis.upload(formdata)
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    originalWavForm = "data:image/png;base64," + data.original_waveform;
                    audioSegments = data.segmented_audios.map(audioBase64 => {
                        return "data:audio/wav;base64," + audioBase64;
                    });
                    wavePlotImages = data.waveform_images.map(imageBase64 => {
                        return "data:image/png;base64," + imageBase64;
                    });
                    audioCount = data.audio_count;
                    audioSegmentedRegions = data.segmented_regions;
                })
                .catch(error => {
                    console.log(error);
                });
        }

        function handleClassification(audioSrc, i) {
            let byteString = atob(audioSrc.split(',')[1]);
            let mimeString = audioSrc.split(',')[0].split(':')[1].split(';')[0];
            let arrayBuffer = new ArrayBuffer(byteString.length);
            let uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < byteString.length; i++) {
                uint8Array[i] = byteString.charCodeAt(i);
            }
            let blob = new Blob([uint8Array], { type: mimeString });
            let file = new File([blob], "audio.wav", { type: blob.type });
            let formData = new FormData();
            formData.append("file", file, file.name);
            console.log('Classifying audio:', formData.get('file'));
            client.AudioAnalysis.classify(formData)
                .then(data => {
                    console.log(data);
                    classification[i] = data.class;
                })
                .catch(error => {
                    console.log(error);
                });
        }

        function handleGenderDetection(audioSrc, i){
            let byteString = atob(audioSrc.split(',')[1]);
            let mimeString = audioSrc.split(',')[0].split(':')[1].split(';')[0];
            let arrayBuffer = new ArrayBuffer(byteString.length);
            let uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < byteString.length; i++) {
                uint8Array[i] = byteString.charCodeAt(i);
            }
            let blob = new Blob([uint8Array], { type: mimeString });
            let file = new File([blob], "audio.wav", { type: blob.type });
            let formData = new FormData();
            formData.append("file", file, file.name);
            client.AudioAnalysis.genderDetection(formData)
                .then(data => {
                    console.log(data);
                    genderDetection[i] = data.prediction;
                    genderPercentage[i] = data.male;
                })
                .catch(error => {
                    console.log(error);
                });
        }

        function handleEmotionDetection(audioSrc, i){
            let byteString = atob(audioSrc.split(',')[1]);
            let mimeString = audioSrc.split(',')[0].split(':')[1].split(';')[0];
            let arrayBuffer = new ArrayBuffer(byteString.length);
            let uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < byteString.length; i++) {
                uint8Array[i] = byteString.charCodeAt(i);
            }
            let blob = new Blob([uint8Array], { type: mimeString });
            let file = new File([blob], "audio.wav", { type: blob.type });
            let formData = new FormData();
            formData.append("file", file, file.name);
            // formData.append("gender", gender);
            client.AudioAnalysis.emotionDetection(formData)
                .then(data => {
                    console.log(data);
                    emotionDetection[i] = data.emotions;
                })
                .catch(error => {
                    console.log(error);
                });
        }

        function handleForensicDetection(audioSrc, i){
            let byteString = atob(audioSrc.split(',')[1]);
            let mimeString = audioSrc.split(',')[0].split(':')[1].split(';')[0];
            let arrayBuffer = new ArrayBuffer(byteString.length);
            let uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < byteString.length; i++) {
                uint8Array[i] = byteString.charCodeAt(i);
            }
            let blob = new Blob([uint8Array], { type: mimeString });
            let file = new File([blob], "audio.wav", { type: blob.type });
            let formData = new FormData();
            formData.append("file", file, file.name);
            client.AudioAnalysis.forensicDetection(formData)
                .then(data => {
                    console.log(data);
                    forensicDetection[i] = data.prediction;
                })
                .catch(error => {
                    console.log(error);
                });
        }


    </script>

<div class="min-h-screen bg-gray-200 py-10">
    <div class="container mx-auto px-4 sm:px-6 lg:px-8">
        <div class="bg-white rounded-lg shadow px-5 py-6 sm:px-6">
            <div class="space-y-6">
                <h1 class="text-center text-2xl font-semibold text-gray-700">Audio Analysis</h1>
                <h2 class="text-center text-xl font-semibold text-gray-700">Upload Audio File (Forensic Aspect: SGM)</h2>

                <div class="space-y-2">
                    <label class="text-sm font-medium text-gray-700" for="file_input">Upload file</label>
                    <input class="w-full text-sm text-gray-700 border border-gray-300 rounded-lg cursor-pointer bg-white focus:outline-none" aria-describedby="file_input_help" type="file" accept="audio/*" on:change={handleFileChange}>
                    <p class="text-sm text-gray-500" id="file_input_help">WAV/Mp3</p>
                </div>

                {#if audioFile}
                <div class="space-y-2">
                    <p>File selected: {audioFile.name}</p>
                    <audio controls>
                        <source src={audioSrc} type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                {/if}

                <div class="flex justify-center">
                    <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleUpload}>Upload</button>
                </div>
            </div>

            <div class="space-y-6">
                <div class="flex justify-center">            
                    <section class="md:w-8/12">
                        {#if originalWavForm}
                            <div class="flex flex-col items-center">
                                <h2 class="text-center mb-4">Input Audio Data (MFDT1)</h2>
                                <img src={originalWavForm} alt="Waveform" class="w-full md:w-3/4 lg:w-1/2 object-contain">
                            </div>
                        {/if}

                        {#if audioCount > 0}
                            <h2 class="text-center text-xl font-semibold text-gray-700">Forensic Aspect DPE</h2>
                            <h2 class="text-center">Total Segments: {audioSegments.length}</h2>
                            <p class="text-center">Data Type : MFDT2 & MFDT3</p>
                            
                            <Carousel>
                                {#each audioSegments as audioSrc, i (i)}
                                <div class="card content-center bg-white rounded-lg shadow-md overflow-hidden mx-auto" style="max-width: 500px;">
                                        <p class="text-xl">Segmentation {i + 1}  {audioSegmentedRegions[i]} </p>
                                        <audio controls>
                                            <source src={audioSrc} type="audio/wav">
                                            Your browser does not support the audio element.
                                        </audio>
                                        
                                        <img src={wavePlotImages[i]} alt="Waveform" class="w-full md:w-1/2 lg:w-1/3 object-contain">

                                        <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleClassification(audioSrc, i)}>Classification</button>
                                        <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleGenderDetection(audioSrc, i)}>Gender Detection</button>
                                        <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleEmotionDetection(audioSrc, i)}>Emotion Detection</button>
                                        <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleForensicDetection(audioSrc, i)}>Forensic Detection</button>
                                        
                                        {#if classification[i]}
                                            <p>Aduio Classification: {classification[i]}, Data : MFDT 4</p>
                                        {/if}
                                        {#if genderDetection[i]}
                                            <p>Gender Classified: {genderDetection[i]}, Male: {genderPercentage[i]}, Female: {100-genderPercentage[i]}, Data : MFDT3 & MFDT 5</p>
                                        {/if}
                                        {#if emotionDetection[i]}
                                            <p>Emotion Classified: {emotionDetection[i]}, Data : MFDT3 & MFDT 5</p>
                                        {/if}
                                        {#if forensicDetection[i]}
                                            <p>Forensic Classified: {forensicDetection[i]}, Data : MFDT3 & MFDT 5</p>
                                        {/if}

                                        
                                        
                                        
                                        
                                    </div>
                                {/each}
                            </Carousel>
                        {:else}
                            <p class="text-center">No audio segments</p>
                            <div class="text-center">
                                <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleClassification(audioSrc, 0)}>Classification Audio</button>
                            </div>
                        {/if}
                    </section>
                </div>
            </div>
        </div>
    </div>
</div>

    <!-- <div class="container mx-auto mt-6">

        <div class="flex justify-center w-full">
            <h1 class="text-2xl font-semibold text-gray-700 dark:text-gray-300">Audio Analysis</h1>
        </div>
        <h1 class="text-start font-head text-2xl font-semibold text-gray-700 dark:text-gray-300">Upload Audio File</h1>
    
        <label class="block mb-2 text-sm font-medium text-gray-900 dark:text-white" for="file_input">Upload file</label>
        <input class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400" aria-describedby="file_input_help" type="file" accept="audio/*" on:change={handleFileChange}>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-300" id="file_input_help">WAV/Mp3</p>
    
        {#if audioFile}
        <p>File selected: {audioFile.name}</p>
            <audio controls>
                <source src={audioSrc} type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        {/if}
    
        <div class="mt-4">
            <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleUpload}>Upload</button>
        </div>

        

        

        

    </div> -->

<style>
    main {
        text-align: center;
        padding: 1em;
        max-width: 400px;
        margin: auto;
    }
    
   

</style>
