<script>
    import axios from 'axios'; // Make sure to install axios using npm install axios
    import client from '../client.js';
    import '../../app.css';

    let audioFile = null;
    let audioSegments = [];
    let audioSrc = null;
    let classification = {};

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
                    
                })
                .catch(error => {
                    console.log(error);
                });
        }

        function handleClassification(audioSrc) {
            console.log('Classifying audio:', audioSrc);
            // Convert base64 to raw binary data held in a string
            let byteString = atob(audioSrc.split(',')[1]);

            // Separate out the mime component
            let mimeString = audioSrc.split(',')[0].split(':')[1].split(';')[0];

            // Write the bytes of the string to an ArrayBuffer
            let arrayBuffer = new ArrayBuffer(byteString.length);
            let uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < byteString.length; i++) {
                uint8Array[i] = byteString.charCodeAt(i);
            }

            // Create a Blob from the ArrayBuffer
            let blob = new Blob([uint8Array], { type: mimeString });

            // Create a File from the Blob
            let file = new File([blob], "audio.wav", { type: blob.type });

            // Create a FormData object
            let formData = new FormData();
            formData.append("file", file, file.name);
            console.log('Classifying audio:', formData.get('file'));
            client.AudioAnalysis.classify(formData)
                .then(data => {
                    console.log(data);
                    classification = data.classification;
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
            {#if audioSegments.length > 0}
                <h2>Segments</h2>
                <p>Click on the audio to play</p>
                
                {#each audioSegments as audioSrc, i (audioSrc)}
                <div class="columns-3 my-3">
                    <audio controls>
                        <source src={audioSrc} type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <p>Audio Segments {i + 1}</p>
                    <button class="rounded bg-emerald-700 px-3 py-1 font-bold text-white hover:bg-emerald-600" on:click={handleClassification(audioSrc)}>Classification</button>
                    <p>The classification result: Gender - {classification}, Emotion - {classification}</p>
                </div>
                {/each}
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
