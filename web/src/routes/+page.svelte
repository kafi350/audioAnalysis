<script>
    import '../app.css';
    import { navigate } from 'svelte-routing';
    
    function goToAudioSection() {
        navigate('/audio');
    }
</script>

<style>
    /* Add your cool design styles here */
</style>

<div>
    <h1>Welcome to Audio Analysis</h1>
    <p>Learn about the fascinating world of audio analysis and its applications.</p>
    
    <!-- Add your logos here -->
    
    <button on:click={goToAudioSection}>Go to Audio Section</button>
</div>
    