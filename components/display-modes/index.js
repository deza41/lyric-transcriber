import ClassicKaraoke from "./ClassicKaraoke";
import MusicVideoLyrics from "./MusicVideoLyrics";
import Teleprompter from "./Teleprompter";
import WaveBounce from "./WaveBounce";
import CenteredSpotlight from "./CenteredSpotlight";

export const DISPLAY_MODES = [
  { id: "classic", label: "Classic", Component: ClassicKaraoke },
  { id: "music-video", label: "Music Video", Component: MusicVideoLyrics },
  { id: "teleprompter", label: "Teleprompter", Component: Teleprompter },
  { id: "wave", label: "Wave", Component: WaveBounce },
  { id: "spotlight", label: "Spotlight", Component: CenteredSpotlight },
];
