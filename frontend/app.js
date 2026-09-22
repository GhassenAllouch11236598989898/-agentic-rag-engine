'use strict';
document.addEventListener('DOMContentLoaded', () => {
  const $ = id => document.getElementById(id);
  const state = {key:'',workspace:null,session:null,busy:false,strategy:'hybrid',controller:null,documents:[],sources:[],checks:[]};
  const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  let toastTimer;
  function toast(text) { $('billing-toast').textContent=text; $('billing-toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('billing-toast').hidden=true,5000); }
  async function api(path, options={}) {
    const headers = new Headers(options.headers || {});
    if(state.key) headers.set('Authorization','Bearer '+state.key);
    if(options.body && !(options.body instanceof FormData)) headers.set('Content-Type','application/json');
    const response=await fetch('/api'+path,{...options,headers});
    if(response.status===401) { switchTab('settings'); throw new Error('Enter your workspace access key in Settings.'); }
    if(!response.ok) {const error=await response.json().catch(()=>({}));throw new Error(typeof error.detail==='string'?error.detail:'Please check your input and try again.');}
    return response;
  }
  function switchTab(name) {
    const aliases={quality:'evaluation',desk:'chat',knowledge:'documents',impact:'overview'};
    name=aliases[name]||name;
    if(!$('panel-'+name)) name='chat';
    document.querySelectorAll('.nav-item').forEach(button=>{button.classList.toggle('active',button.dataset.tab===name);button.setAttribute('aria-selected',button.dataset.tab===name);});
    document.querySelectorAll('.tab-panel').forEach(panel=>panel.classList.toggle('active',panel.id==='panel-'+name));
    history.replaceState(null,'','#'+name);
    if(name==='documents' && state.workspace) loadDocuments().catch(error=>toast(error.message));
    if(name==='overview' && state.workspace) refreshActivity().catch(error=>toast(error.message));
  }
  document.querySelectorAll('.nav-item').forEach(button=>button.addEventListener('click',()=>switchTab(button.dataset.tab)));
  function welcome() {
    state.controller?.abort();state.controller=null;state.busy=false;state.session=null;state.sources=[];$('send-btn').disabled=false;
    $('chat-messages').innerHTML='<div class="billing-welcome"><div class="billing-welcome-icon">✧</div><h3>Billing questions. Clear, sourced answers.</h3><p>Draft a reply using approved policies, inspect the evidence, and review it before sharing.</p><div class="billing-suggestions" id="billing-suggestions"></div><p class="billing-welcome-note">No payments are processed. No refunds or messages are sent.</p></div>';
    const tickets=state.workspace?.tickets || [];
    $('billing-suggestions').innerHTML=tickets.map((ticket,index)=>'<button class="secondary-btn" data-prompt="'+index+'">'+escapeHtml(ticket.subject)+' <span>↗</span></button>').join('');
    $('billing-suggestions').querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>send(tickets[Number(button.dataset.prompt)].message)));
    renderSources([]);
  }
  function renderSources(sources) {
    state.sources=sources;
    $('source-count').textContent=sources.length+' sources';
    $('passages-list').innerHTML=sources.length?sources.map((source,index)=>'<div class="passage-card" id="passage-card-'+source.citation+'"><div class="passage-top-row"><span class="passage-num">'+String(index+1).padStart(2,'0')+'</span><div class="passage-source-pill"><span class="source-file-badge pdf small">doc</span><span class="passage-file-name">'+escapeHtml(source.title)+'</span><span class="passage-page-tag">'+(source.page?'Page '+source.page:'Passage')+'</span></div></div><p class="passage-quote">'+escapeHtml(source.content)+'</p><button class="link-btn" data-document="'+escapeHtml(source.document_id)+'">Open original document →</button></div>').join(''):'<p class="billing-empty">No evidence retrieved yet. Ask a billing question to inspect its policy sources.</p>';
    $('passages-list').querySelectorAll('[data-document]').forEach(button=>button.addEventListener('click',()=>openDocument(button.dataset.document)));
  }
  function renderText(element,text,sources) {
    const valid=new Set(sources.map(source=>source.citation));
    element.innerHTML=escapeHtml(text).replace(/\[(\d+)\]/g,(match,id)=>valid.has(Number(id))?'<button class="citation-chip" data-citation="'+id+'" aria-label="Inspect source '+id+'">['+id+']</button>':match);
    element.querySelectorAll('[data-citation]').forEach(button=>button.addEventListener('click',()=>{
      renderSources(sources);
      const card=$('passage-card-'+button.dataset.citation);
      if(card) {card.classList.add('billing-highlight');card.scrollIntoView({behavior:'smooth',block:'nearest'});}
    }));
  }
  async function send(question) {
    if(state.busy) return;
    question=(question || $('chat-input').value).trim();
    if(question.length<2) {toast('Enter a billing question first.');return;}
    if(question.length>6000) {toast('Please keep the question under 6,000 characters.');return;}
    $('chat-messages').querySelector('.billing-welcome')?.remove();
    $('chat-input').value='';state.busy=true;$('send-btn').disabled=true;
    const controller=new AbortController();state.controller=controller;
    const user=document.createElement('div');user.className='message-row user-row';
    user.innerHTML='<div class="message-bubble user-bubble"><p>'+escapeHtml(question)+'</p></div><div class="user-avatar-badge">U</div>';
    $('chat-messages').appendChild(user);
    const process=document.createElement('div');process.className='agent-process-bar';
    process.innerHTML='<div class="process-steps"><span class="process-step">Searching billing policies…</span></div><div class="process-elapsed">Preparing draft</div>';
    $('chat-messages').appendChild(process);
    const row=document.createElement('div');row.className='message-row assistant-row';
    row.innerHTML='<div class="bot-avatar-badge">✧</div><div class="message-bubble assistant-bubble"><div class="bubble-content billing-answer">Finding relevant evidence…</div></div>';
    $('chat-messages').appendChild(row);
    const bubble=row.querySelector('.message-bubble'),content=row.querySelector('.bubble-content');
    $('chat-messages').scrollTop=$('chat-messages').scrollHeight;
    try {
      const response=await api('/chat/stream',{method:'POST',signal:controller.signal,body:JSON.stringify({message:question,session_id:state.session,search_type:state.strategy==='semantic'?'vector':'hybrid'})});
      const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',completed=false;
      while(true) {
        const {done,value}=await reader.read();if(done)break;
        buffer+=decoder.decode(value,{stream:true});let boundary;
        while((boundary=buffer.indexOf('\n\n'))!==-1) {
          const block=buffer.slice(0,boundary);buffer=buffer.slice(boundary+2);
          const line=block.split('\n').find(line=>line.startsWith('data: '));if(!line)continue;
          const event=JSON.parse(line.slice(6));
          if(event.type==='error')throw new Error(event.message);
          if(event.type==='progress')process.querySelector('.process-step').textContent=event.stage;
          if(event.type==='result') {
            if(state.controller!==controller)return;
            completed=true;state.session=event.session_id;
            renderText(content,event.message,event.sources);renderSources(event.sources);
            const escalation=event.status==='needs_escalation';
            process.querySelector('.process-step').textContent=escalation?'Specialist review needed':'✓ Policies searched  →  ✓ References checked';
            process.querySelector('.process-elapsed').textContent=event.method+' · '+(event.elapsed_ms/1000).toFixed(1)+'s';
            const meta=document.createElement('div');meta.className='billing-draft-note';
            meta.textContent=escalation?'This request needs a billing specialist. Account-specific facts require a billing-system lookup.':'Draft for human review. Valid references do not guarantee every claim is correct.';
            bubble.appendChild(meta);
            if(event.sources.length) {
              const sourceArea=document.createElement('div');sourceArea.className='bubble-sources-section';
              sourceArea.innerHTML='<div class="sources-label-row">Sources</div><div class="sources-cards-row">'+event.sources.map(source=>'<button class="source-card" data-document="'+escapeHtml(source.document_id)+'"><div class="source-file-badge pdf">doc</div><div class="source-info"><div class="source-filename">'+escapeHtml(source.title)+'</div><div class="source-page">'+(source.page?'Page '+source.page:'Indexed passage')+'</div></div><div class="source-chevron">›</div></button>').join('')+'</div>';
              bubble.appendChild(sourceArea);sourceArea.querySelectorAll('[data-document]').forEach(button=>button.addEventListener('click',()=>openDocument(button.dataset.document)));
            }
            const actions=document.createElement('div');actions.className='billing-draft-actions';
            actions.innerHTML='<button class="secondary-btn" data-action="copy">'+(escalation?'Copy handoff':'Copy reply')+'</button><button class="link-btn" data-action="edit">Edit draft</button><button class="link-btn" data-action="helpful">Helpful</button><button class="link-btn" data-action="needs_work">Needs work</button>';
            bubble.appendChild(actions);
            let reply=event.message,editor=null;
            actions.querySelector('[data-action="copy"]').addEventListener('click',async()=>{
              try {await navigator.clipboard.writeText(editor?editor.value:reply);toast('Copied. Review the wording before sharing.');}catch{toast('Select the reply text and copy it manually.');}
            });
            actions.querySelector('[data-action="edit"]').addEventListener('click',e=>{
              if(editor) {reply=editor.value;editor.remove();editor=null;content.hidden=false;renderText(content,reply,event.sources);e.target.textContent='Edit draft';meta.textContent='You edited this draft. Recheck claims and references before sharing.';}
              else {editor=document.createElement('textarea');editor.className='billing-editor';editor.value=reply;editor.setAttribute('aria-label','Edit billing reply');content.hidden=true;content.after(editor);e.target.textContent='Save edits';editor.focus();}
            });
            ['helpful','needs_work'].forEach(value=>actions.querySelector('[data-action="'+value+'"]').addEventListener('click',async()=>{
              try {await api('/drafts/'+event.draft_id+'/feedback',{method:'POST',body:JSON.stringify({value})});toast('Feedback saved.');await refreshActivity();}catch(error){toast(error.message);}
            }));
          }
        }
      }
      if(!completed)throw new Error('The connection ended before a reply was ready. Please try again.');
      refreshActivity().catch(error=>toast(error.message));
    } catch(error) {
      if(error.name!=='AbortError' && state.controller===controller) {content.textContent=error.message;content.classList.add('billing-error');process.querySelector('.process-step').textContent='Request could not be completed';process.querySelector('.process-elapsed').textContent='Please retry';}
    } finally {
      if(state.controller===controller) {state.busy=false;state.controller=null;$('send-btn').disabled=false;$('chat-messages').scrollTop=$('chat-messages').scrollHeight;}
    }
  }
  async function loadDocuments() {
    state.documents=(await (await api('/documents')).json()).documents;
    $('kpi-docs').textContent=state.documents.length;$('kpi-chunks').textContent=state.documents.reduce((sum,doc)=>sum+(doc.chunk_count||0),0);
    $('documents-grid').innerHTML=state.documents.map(doc=>'<div class="doc-card"><div class="doc-card-title"><span class="source-file-badge pdf">doc</span>'+escapeHtml(doc.title)+'</div><div class="doc-card-meta">'+(doc.chunk_count||0)+' passages · '+(doc.kind==='sample'?'Fictional sample policy':'Uploaded document')+'</div><div class="billing-draft-actions"><button class="link-btn" data-document="'+escapeHtml(doc.id)+'">Open document →</button>'+(state.workspace?.role==='admin'?'<button class="link-btn billing-remove" data-delete="'+escapeHtml(doc.id)+'">Remove</button>':'')+'</div></div>').join('') || '<p class="billing-empty">No billing documents yet. Upload a policy to get started.</p>';
    $('documents-grid').querySelectorAll('[data-document]').forEach(button=>button.addEventListener('click',()=>openDocument(button.dataset.document)));
    $('documents-grid').querySelectorAll('[data-delete]').forEach(button=>button.addEventListener('click',async()=>{
      const doc=state.documents.find(doc=>doc.id===button.dataset.delete);
      if(!confirm('Remove '+doc.title+' from the knowledge library? It will no longer be available for new replies.'))return;
      try {await api('/documents/'+encodeURIComponent(doc.id),{method:'DELETE'});await loadDocuments();toast('Document removed.');}catch(error){toast(error.message);}
    }));
  }
  async function openDocument(id) {
    $('document-title').textContent='Loading source…';$('document-content').textContent='';if(!$('document-dialog').open)$('document-dialog').showModal();
    try {const doc=await(await api('/documents/'+encodeURIComponent(id))).json();$('document-title').textContent=doc.title;$('document-content').textContent=doc.content;}
    catch(error){$('document-content').textContent=error.message;}
  }
  async function refreshActivity() {
    const {drafts}=await(await api('/activity')).json();$('kpi-drafts').textContent=drafts.length;$('kpi-helpful').textContent=drafts.filter(draft=>draft.feedback==='helpful').length;
  }
  function openUpload() {
    if(state.workspace?.role!=='admin'){toast('A workspace administrator must upload documents.');return;}
    $('file-input').value='';$('upload-status').textContent='';$('upload-modal').style.display='flex';
  }
  async function upload() {
    const file=$('file-input').files[0];if(!file){$('upload-status').textContent='Choose a PDF, Markdown, or text file first.';return;}
    if(file.size>10*1024*1024){$('upload-status').textContent='Files must be no larger than 10 MB.';return;}
    const form=new FormData();form.append('file',file);$('modal-submit-btn').disabled=true;$('upload-status').textContent='Reading and indexing your document…';
    try {const result=await(await api('/documents',{method:'POST',body:form})).json();await loadDocuments();$('upload-modal').style.display='none';toast(result.duplicate?'This document is already indexed.':'Document indexed. Ask a question about it now.');}
    catch(error){$('upload-status').textContent=error.message;}
    finally{$('modal-submit-btn').disabled=false;}
  }
  async function search() {
    const query=$('playground-query').value.trim();if(query.length<2)return;
    const strategy=document.querySelector('input[name="search-strat"]:checked').value;
    $('playground-search-btn').disabled=true;$('playground-results').textContent='Searching billing policies…';
    try {const {results}=await(await api('/search/'+strategy,{method:'POST',body:JSON.stringify({query})})).json();
      $('playground-results').innerHTML=results.map(source=>'<div class="passage-card"><div class="passage-top-row"><strong>'+escapeHtml(source.title)+'</strong></div><p class="passage-quote">'+escapeHtml(source.content)+'</p><button class="link-btn" data-document="'+escapeHtml(source.document_id)+'">Open document →</button></div>').join('')||'<p class="billing-empty">No sufficiently relevant passages were found.</p>';
      $('playground-results').querySelectorAll('[data-document]').forEach(button=>button.addEventListener('click',()=>openDocument(button.dataset.document)));
    }catch(error){$('playground-results').textContent=error.message;}finally{$('playground-search-btn').disabled=false;}
  }
  async function runChecks() {
    $('btn-run-evaluation').disabled=true;$('check-summary').textContent='Running actual billing checks…';
    try {
      const result=await(await api('/benchmark/run',{method:'POST'})).json();state.checks=result.cases;
      $('check-summary').textContent=result.passed+' / '+result.total+' passed · '+new Date(result.run_at).toLocaleTimeString();
      $('check-rows').innerHTML=result.cases.map((item,index)=>'<tr><td><button class="link-btn" data-check="'+index+'">'+escapeHtml(item.name)+'</button></td><td>'+ (item.retrieval_ok?'✓ Pass':'Needs review')+'</td><td>'+(item.references_ok?'✓ Pass / N/A':'Needs review')+'</td><td>'+(item.behavior_ok?'✓ Pass':'Needs review')+'</td><td><span class="billing-status '+(item.passed?'':'warning')+'">'+(item.passed?'Passed':'Needs attention')+'</span></td></tr>').join('');
      $('check-rows').querySelectorAll('[data-check]').forEach(button=>button.addEventListener('click',()=>showCheck(Number(button.dataset.check))));
      showCheck(0);
    }catch(error){$('check-summary').textContent=error.message;}finally{$('btn-run-evaluation').disabled=false;}
  }
  function showCheck(index) {
    const item=state.checks[index];if(!item)return;
    $('check-inspector').innerHTML='<h4>'+escapeHtml(item.question)+'</h4><p class="billing-check-answer">'+escapeHtml(item.message)+'</p><p class="header-sub-text">Expected policy: '+escapeHtml(item.expected_source||'No source required; escalate')+'</p><p class="header-sub-text">Retrieved: '+escapeHtml(item.actual_sources.join(', ')||'No matching evidence')+'</p>';
  }
  function updateImpact() {
    const read=id=>Math.max(0,Math.min(Number($(id).max),Number($(id).value)||0));
    const hours=read('roi-tickets')*read('roi-minutes')*read('roi-share')/100/60;
    $('roi-hours').textContent=Math.round(hours).toLocaleString()+' hours';
    $('roi-value').textContent=new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(hours*read('roi-cost'))+' estimated capacity value / month';
  }
  async function initialize() {
    try {
      state.workspace=await(await api('/workspace')).json();
      const demo=state.workspace.mode==='demo';
      $('active-model-name').textContent=demo?'Local billing demo':'Connected AI workspace';
      $('status-mode-label').textContent=demo?'Local demo':'Connected mode';
      $('mode-banner').textContent=demo?'Interactive demo · Fictional billing policies · Real retrieval · Extractive replies':'Connected billing workspace · AI-assisted drafts · Review every reply before sharing';
      $('workspace-display').value=state.workspace.name;$('mode-display').value=demo?'Local keyword retrieval + extractive preview':'PostgreSQL retrieval + configured AI provider';$('role-display').value=state.workspace.role;
      $('upload-doc-modal-btn').hidden=state.workspace.role!=='admin';$('chat-attach-btn').hidden=state.workspace.role!=='admin';
      document.querySelectorAll('.segment-btn').forEach(button=>{button.disabled=demo?button.dataset.strategy!=='keyword':button.dataset.strategy==='keyword';button.classList.toggle('active',demo?button.dataset.strategy==='keyword':button.dataset.strategy==='hybrid');});
      state.strategy=demo?'keyword':'hybrid';
      document.querySelector('input[name="search-strat"][value="vector"]').disabled=demo;
      document.querySelector('.strategy-options').title=demo?'Local demo uses keyword retrieval. Connected mode enables vector search and hybrid RRF.':'Choose your retrieval method';
      document.querySelector('input[name="search-strat"][value="hybrid"]').parentElement.lastChild.textContent=demo?' Local keyword':' Hybrid (RRF)';
      $('btn-run-evaluation').disabled=!demo;
      if(!demo)$('check-summary').textContent='Create billing checks from your client policies before evaluating this connected workspace.';
      welcome();await Promise.all([loadDocuments(),refreshActivity()]);return true;
    }catch(error){$('active-model-name').textContent='Connection needed';toast(error.message);return false;}
  }
  $('send-btn').addEventListener('click',()=>send());
  $('chat-input').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}});
  $('chat-options-btn').addEventListener('click',welcome);
  $('upload-doc-modal-btn').addEventListener('click',openUpload);$('chat-attach-btn').addEventListener('click',openUpload);
  $('modal-close-btn').addEventListener('click',()=>$('upload-modal').style.display='none');$('modal-cancel-btn').addEventListener('click',()=>$('upload-modal').style.display='none');
  $('modal-submit-btn').addEventListener('click',upload);
  $('file-input').addEventListener('change',()=>{$('upload-status').textContent=$('file-input').files[0]?.name||'';});
  ['dragenter','dragover'].forEach(type=>$('drop-zone').addEventListener(type,event=>{event.preventDefault();$('drop-zone').classList.add('billing-highlight');}));
  ['dragleave','drop'].forEach(type=>$('drop-zone').addEventListener(type,event=>{event.preventDefault();$('drop-zone').classList.remove('billing-highlight');}));
  $('drop-zone').addEventListener('drop',event=>{if(event.dataTransfer.files.length){const transfer=new DataTransfer();transfer.items.add(event.dataTransfer.files[0]);$('file-input').files=transfer.files;$('upload-status').textContent=transfer.files[0].name;}});
  $('close-source').addEventListener('click',()=>$('document-dialog').close());
  $('refresh-docs-btn').addEventListener('click',()=>loadDocuments().catch(error=>toast(error.message)));
  $('playground-search-btn').addEventListener('click',search);$('playground-query').addEventListener('keydown',event=>{if(event.key==='Enter')search();});
  $('btn-run-evaluation').addEventListener('click',runChecks);
  document.querySelectorAll('.segment-btn').forEach(button=>button.addEventListener('click',()=>{state.strategy=button.dataset.strategy;document.querySelectorAll('.segment-btn').forEach(item=>item.classList.toggle('active',item===button));}));
  $('access-form').addEventListener('submit',async event=>{event.preventDefault();state.key=$('access-key').value;$('access-key').value='';$('access-error').textContent='';if(await initialize()){switchTab('chat');toast('Workspace connected.');}else $('access-error').textContent='Could not connect. Check your key and try again.';});
  ['roi-tickets','roi-minutes','roi-share','roi-cost'].forEach(id=>$(id).addEventListener('input',updateImpact));
  switchTab(location.hash.slice(1)||'chat');updateImpact();initialize();
});
