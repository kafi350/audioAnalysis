<script>
    import axios from 'axios'; // Make sure to install axios using npm install axios
    import client from '../client.js';
    import '../../app.css';

    let audioFile = null;
    let audioSegments = [];
    let audioCount = 0;
    let audioSrc = null;
    let classification = Array(audioSegments.length).fill('');
    let genderDetection = Array(audioSegments.length).fill('');
    let genderPercentage = Array(audioSegments.length).fill('');

    function handleFileChange(event) {
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
                    audioSegments = data.segmented_audios.map(audioBase64 => {
                        return "data:audio/wav;base64," + audioBase64;
                    });
                    audioCount = data.audio_count;
                    
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
                    // classification[i] = data.prediction;
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
                    // classification[i] = data.prediction;
                })
                .catch(error => {
                    console.log(error);
                });
        }


    </script>

    <div class="container mx-auto mt-6">
        <h1 class="text-start font-head text-2xl font-semibold text-gray-700 dark:text-gray-300">Upload Audio File</h1>
        
        <label class="block mb-2 text-sm font-medium text-gray-900 dark:text-white" for="file_input">Upload file</label>
        <input class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400" aria-describedby="file_input_help" type="file" accept="audio/*" on:change={handleFileChange}>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-300" id="file_input_help">WAV/Mp3</p>

        <!-- <input type="file" accept="audio/*" on:change={handleFileChange}/> -->
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

        

        <section>
            {#if audioCount > 0}
                <h2>Segments</h2>
                <p>Click on the audio to play</p>
                
                {#each audioSegments as audioSrc, i (audioSrc)}
                <div class="columns-3 my-3">
                    <audio controls>
                        <source src={audioSrc} type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <p>Audio Segments {i + 1}</p>
                    <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleClassification(audioSrc, i)}>Classification</button>
                    <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleGenderDetection(audioSrc, i)}>Gender Detection</button>
                    <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleEmotionDetection(audioSrc, i)}>Emotion Detection</button>
                    <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleForensicDetection(audioSrc, i)}>Forensic Detection</button>
                    <p>The classification result: {classification[i]}</p>
                    <p>Gender Classified: {genderDetection[i]}, Male: {genderPercentage[i]}, Female: {100-genderPercentage[i]}</p>
                </div>
                {/each}
            {:else}
                <p>No audio segments</p>
                <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleClassification(audioSrc, 0)}>Classification Audio</button>
            {/if}

            
        </section>

    </div>

<style>
    main {
        text-align: center;
        padding: 1em;
        max-width: 400px;
        margin: auto;
    }
    input {
        margin-top: 1em;
    }
</style>
