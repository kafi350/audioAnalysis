<script>
    import axios from 'axios'; // Make sure to install axios using npm install axios
    import client from '../client.js';

    let audioFile = null;
    let audioSegments = [];
    let audioSrc = null;

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
    </script>

    <main>
        <h1>Upload Audio File</h1>
        <input type="file" accept="audio/*" on:change={handleFileChange}/>
        {#if audioFile}
        <p>File selected: {audioFile.name}</p>
            <audio controls>
                <source src={audioSrc} type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        {/if}
        <button on:click={handleUpload}>Upload</button>
        <section>
            {#if audioSegments.length > 0}
                <h2>Segments</h2>
                <p>Click on the audio to play</p>
                
                {#each audioSegments as audioSrc (audioSrc)}
                    <audio controls>
                        <source src={audioSrc} type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                {/each}
            {/if}

            <button on:click={() => {window.location.href = '/audio/analysis'}}>Classification</button>
        </section>

    </main>

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
